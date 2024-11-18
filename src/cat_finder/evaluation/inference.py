import math
import torch
from torch_geometric.utils import to_dense_batch


class InfDicts(object):
    def __init__(self, truth, pred):
        """Create inference for object condensation prediction
        Class creates the inference dictionary depending on the type net trained and the beta mask

        Args:
            truth (torch tensor): Batch of truth values with dimensions [batch, pixels, values]
            pred (torch tensor): Batch of predicted values with dimensions [batch, pixels, values]
            type (str, optional): One of 'gnn', 'image', 'cnn', type of net to evaluate. Defaults to 'gnn'.
        """
        self.truth = truth
        self.pred = pred
        self.type = type

    def get_infdict(self, graph=True, coord_dim=2):

        if graph:
            self.dense_truth, self.truth_mask = to_dense_batch(
                self.truth.y, batch=self.truth.batch
            )
            self.dense_pred, self.pred_mask = to_dense_batch(
                self.pred, batch=self.truth.batch
            )
        else:
            self.dense_truth = self.truth
            self.dense_pred = self.pred

        outdict = {}
        outdict["t_mask"] = self.dense_truth[:, :, 0]  # is this noise or not
        outdict["t_objidx"] = self.dense_truth[:, :, 1]  # which object is this
        outdict["truth"] = self.dense_truth[:, :, 2:]  # nD momentum

        outdict["p_beta"] = self.dense_pred[:, :, 0]  # predicted beta score
        outdict["p_ccoords"] = self.dense_pred[
            :, :, 1 : 1 + coord_dim
        ]  # predicted condensation point coordinates
        outdict["p_prediction"] = self.dense_pred[
            :, :, 1 + coord_dim :
        ]  # track parameter predictions

        self.inf_dict = outdict

        return self.inf_dict

    @staticmethod
    def get_distance(point_candidate, beta_selected, t_distance):
        for point in beta_selected:
            distances = torch.sqrt(torch.sum((point_candidate - point) ** 2))
            if distances <= t_distance:
                return False
        return True

    @staticmethod
    def get_alldistance(candidate_ccoords, all_ccoords, t_distance):
        difference = all_ccoords - candidate_ccoords
        distances = torch.linalg.norm(difference)
        return distances > t_distance

    def collect_over_threshold(self, ccoords, beta_idxs, beta_preselected, t_distance):
        """Iterate through pre selected beta values and only return those with a distance
        in the coordinate space more than the threshold to other beta values

        Args:
            ccoords (torch.array): two-dimensional array containing predicted x,y center values
            beta_sorted (torch.array): beta values in descending order
            beta_idxs (torch.array): corresponding indices to beta sorted
            beta_preselected (torch.array): True/False array depending on whether beta is higher than threshold
            t_distance (float): distance threshold

        Returns:
            torch.array: Array, where only beta values after selection are true
        """

        cond_points = [ccoords[beta_idxs[0]]]

        for idx in beta_idxs[1:]:
            if beta_preselected[idx]:
                cond_candidate = ccoords[idx]
                candidate = self.get_distance(cond_candidate, cond_points, t_distance)
                if candidate:
                    cond_points.append(cond_candidate)
                else:
                    beta_preselected[idx] = False
        return beta_preselected

    def get_betamask(self, t_beta, t_distance):
        """Do inference on the given data, by selecting only beta values with values over t_beta and with a distance
        higher than t_distance between those values.

        Args:
            data (torch.array): Dictionary of predicted values
            t_beta (float): beta threshold
            t_distance (float): distance threshold

        Returns:
            torch.array: Array, where only beta values after selection are true
        """

        predicted_betas = torch.reshape(
            self.inf_dict["p_beta"], (self.inf_dict["p_beta"].size(dim=0), -1)
        )
        ccoords = torch.reshape(
            self.inf_dict["p_ccoords"],
            (
                self.inf_dict["p_ccoords"].size(dim=0),
                -1,
                self.inf_dict["p_ccoords"].size(dim=-1),
            ),
        )

        beta_sorted, beta_idxs = torch.sort(predicted_betas, dim=1, descending=True)
        beta_preselected = predicted_betas > t_beta
        beta_selected = torch.zeros_like(beta_preselected)

        for i in range(len(predicted_betas)):
            beta_selected[i] = self.collect_over_threshold(
                ccoords[i], beta_idxs[i], beta_preselected[i], t_distance
            )

        self.beta_mask = torch.reshape(beta_selected, self.inf_dict["p_beta"].size())

        return self.beta_mask
