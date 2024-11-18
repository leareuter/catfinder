import torch
import numpy as np
from torch_geometric.utils import to_dense_batch

eps = 1e-7


def average_gnn_loss(
    truth,
    pred,
    coord_dim=2,
    device="cuda",
):
    """Generate same sized torch.Tensors from graphs and calculate loss

    Args:
        truth (Graph object): Truth graph
        pred (Graph object): Predicted graph

    Returns:
        floatx6, torch.Tensorx4: loss values and dense truth and predictions
        with masks
    """
    dense_truth, truth_mask = to_dense_batch(truth.y, batch=truth.batch)
    dense_pred, pred_mask = to_dense_batch(pred, batch=truth.batch)

    (
        reploss,
        attloss,
        betaloss,
        suppress_noise_loss,
        pl,
        vertexl,
        chargel,
    ) = gnn_condensation_loss(
        dense_truth,
        dense_pred,
        device=device,
        coord_dim=coord_dim,
    )
    return (
        reploss,
        attloss,
        betaloss,
        suppress_noise_loss,
        pl,
        vertexl,
        chargel,
        dense_truth,
        dense_pred,
        truth_mask,
        pred_mask,
    )


def create_gnn_dict(truth, pred, coord_dim=2):
    """Create loss dictionary from truth and predicted values

    Args:
        truth (torch.Tensor): Input truth tensor with dimensions crystals x batch_size x truth values
        pred (torch.Tensor): Input prediction tensor with dimensions crystals x batch_size x predicted values

    Returns:
        Dict: Dictionary containing true and predicted values as torch.Tensors, including noise and active points
    """

    outdict = {}
    outdict["t_mask"] = truth[
        :, :, 0:1
    ]  # this is binary signal or background truth information
    outdict["t_objidx"] = truth[
        :, :, 1:2
    ]  # truth level which object (0 for background and 1 to max_n_objects for objects)

    outdict["p_beta"] = pred[:, :, 0:1]  # predicted beta value
    outdict["p_ccoords"] = pred[
        :, :, 1 : 1 + coord_dim
    ]  # predicted latent space coordinates

    outdict["t_p"] = truth[:, :, 2:5]  # Truth momentum vector
    outdict["p_p"] = pred[
        :, :, 1 + coord_dim : 1 + coord_dim + 3
    ]  # predicted momentum vector

    outdict["t_vertex"] = truth[:, :, 5:8]  # Truth track starting position
    outdict["p_vertex"] = pred[
        :, :, 1 + coord_dim + 3 : 1 + coord_dim + 2 * 3
    ]  # predicted track starting position

    outdict["t_charge"] = truth[:, :, 8:9]  # Truth charge, 1 if positive, 0 if negative
    outdict["p_charge"] = pred[
        :, :, 1 + coord_dim + 2 * 3 : 1 + coord_dim + 2 * 3 + 1
    ]  # predicted charge, 1 if positive, 0 if negative

    flattened = torch.reshape(outdict["t_mask"], (outdict["t_mask"].size()[0], -1))
    outdict["n_nonoise"] = torch.unsqueeze(
        torch.count_nonzero(flattened, dim=-1).type(torch.FloatTensor), axis=1
    )  # number of signal points

    outdict["n_noise"] = (
        float(truth.shape[1]) - outdict["n_nonoise"]
    )  # number of background points

    return outdict


def beta_weighted_truth_mean(l_in, d, beta_scaling):
    """Weigh the loss(pred, truth) with the beta prediction, so correct parameters are
    predicted onto condensation point

    Args:
        l_in (torch.array): Difference between predicted and true values
        d (dictionary): dictionary of predicted and true values
        beta_scaling (torch.array): charge of every point

    Returns:
        torch.array: loss weighted with beta values
    """
    l_in = torch.sum(beta_scaling * d["t_mask"] * l_in, axis=1)
    den = torch.sum(d["t_mask"] * beta_scaling, axis=1) + eps
    return l_in / den


def p_loss(d, beta_scaling):
    pl = torch.sum(torch.abs(d["t_p"] - d["p_p"]), axis=-1, keepdims=True)
    return beta_weighted_truth_mean(pl, d, beta_scaling)


def vertex_loss(d, beta_scaling):
    pvertex = torch.sum(
        torch.abs(d["t_vertex"] - d["p_vertex"]), axis=-1, keepdims=True
    )
    return beta_weighted_truth_mean(pvertex, d, beta_scaling)


def charge_cross_entr_loss(d, beta_scaling):
    tID = d["t_mask"] * d["t_charge"]
    tID = torch.where(tID <= 0.0, torch.zeros_like(tID) + 10 * eps, tID)
    tID = torch.where(tID >= 1.0, torch.zeros_like(tID) + 1.0 - 10 * eps, tID)

    pID = d["t_mask"] * d["p_charge"]
    pID = torch.where(pID <= 0.0, torch.zeros_like(tID) + 10 * eps, pID)
    pID = torch.where(pID >= 1.0, torch.zeros_like(tID) + 1.0 - 10 * eps, pID)

    xentr = (
        (-1.0) * d["t_mask"] * (tID * torch.log(pID) + (1 - tID) * torch.log(1 - pID))
    )

    beta_scaled_loss = beta_weighted_truth_mean(xentr, d, beta_scaling)
    return beta_scaled_loss


def calculate_charge(beta, q_min):
    beta = torch.clamp(beta, 0, 1 - eps)
    return torch.atanh(beta) + q_min


def gnn_condensation_loss(
    truth,
    pred,
    device="cuda",
    coord_dim=2,
):
    d = create_gnn_dict(truth, pred, coord_dim=coord_dim)

    reploss, attloss, betaloss, suppress_noise_loss = sub_gnn_condensation_loss(
        d,
        q_min=1.0,
        Ntotal=truth.shape[1],
        device=device,
    )

    payload_scaling = calculate_charge(d["p_beta"], 0.1)

    pl = torch.mean(p_loss(d, payload_scaling))

    vertexl = torch.mean(vertex_loss(d, payload_scaling))
    chargel = torch.mean(charge_cross_entr_loss(d, payload_scaling))
    return reploss, attloss, betaloss, suppress_noise_loss, pl, vertexl, chargel


def sub_gnn_condensation_loss(d, q_min, Ntotal=4096, device="cuda", max_n_objects=6):
    q = calculate_charge(d["p_beta"], q_min)

    L_att = torch.zeros_like(q[:, 0, 0])
    L_rep = torch.zeros_like(q[:, 0, 0])
    L_beta = torch.zeros_like(q[:, 0, 0])

    Nobj = torch.zeros_like(q[:, 0, 0])
    isobj = []
    alpha = []

    for k in range(1, int(torch.max(d["t_objidx"])) + 1):  # loop over number of objects
        Mki = torch.where(
            torch.abs(d["t_objidx"] - float(k)) < 0.2,
            torch.zeros_like(q) + 1,
            torch.zeros_like(q),
        )

        iobj_k, _ = torch.max(Mki, dim=1)
        Nobj += torch.squeeze(iobj_k, dim=1)

        kalpha = torch.argmax(Mki * d["t_mask"] * d["p_beta"], dim=1)

        isobj.append(iobj_k)
        alpha.append(kalpha)

        dim0_tensor = torch.arange(0, kalpha.size()[0]).type(torch.long)
        x_kalpha = d["p_ccoords"][
            dim0_tensor, torch.squeeze(kalpha, dim=1), :
        ]  # translation of tf.gather_nd(batch_dims=1)
        x_kalpha = torch.unsqueeze(x_kalpha, axis=1)

        q_kalpha = q[dim0_tensor, torch.squeeze(kalpha, dim=1), :]
        q_kalpha = torch.unsqueeze(q_kalpha, axis=1)
        distance = torch.sqrt(
            torch.sum((x_kalpha - d["p_ccoords"]) ** 2, dim=-1, keepdims=True) + eps
        )

        F_att = q_kalpha * q * distance**2 * Mki
        F_rep = q_kalpha * q * torch.nn.functional.relu(1.0 - distance) * (1.0 - Mki)

        L_att += (
            torch.squeeze(iobj_k * torch.sum(F_att, axis=1), axis=1) / Ntotal
        )  # attract objects to this object
        L_rep += (
            torch.squeeze(iobj_k * torch.sum(F_rep, axis=1), axis=1) / Ntotal
        )  # repulse points from this

        beta_kalpha = d["p_beta"][dim0_tensor, torch.squeeze(kalpha, dim=1), :]
        L_beta += torch.squeeze(iobj_k * (1 - beta_kalpha), axis=1)

    L_beta /= Nobj
    L_suppnoise = torch.squeeze(
        torch.sum((1.0 - d["t_mask"]) * d["p_beta"], axis=1).to(device)
        / (d["n_noise"] + eps).to(device),
        axis=1,
    )

    reploss = torch.mean(L_rep)
    attloss = torch.mean(L_att)
    betaloss = torch.mean(L_beta)
    suppress_noise_loss = torch.mean(L_suppnoise)

    return reploss, attloss, betaloss, suppress_noise_loss
