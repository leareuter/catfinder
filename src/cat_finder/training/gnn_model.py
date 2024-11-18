import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GravNetConv, BatchNorm, global_mean_pool


"""
Based on:
Object condensation: 
one-stage grid-free multi-object reconstruction in physics detectors, graph and image data
https://arxiv.org/abs/2002.03605

Isabel Haide
Improving ECL Clustering on Trigger Level with Object Condensation 
"""


class CDCNet(nn.Module):
    def __init__(
        self,
        input_dim,
        k=10,
        dim1=64,
        dim2=32,
        nblocks=4,
        coord_dim=2,
        space_dimensions=4,
        momentum=0.6,
    ):
        """
        Modular CDCNet for Object Condensation,
        number of input features, hidden dimensions, and blocks are adjustable

        input dim (int):    number of input features per node
        k (int):            number of k-nearest neighbours for GravNetConv
        nblocks:            number of blocks the network consists
        coord_dim:          dimension for object condensation coordinates
        space_dimensions:   number of space dimensions for GravNetConv
        momentum:           momentum for batch normalization

        GravNet from:
        Learning Representations of Irregular
        Particle-detector Geometry with Distance-weighted Graph
        Networks"
        https://arxiv.org/abs/1902.07987
        """
        super().__init__()

        self.batch_norm_0 = BatchNorm(input_dim, momentum=0.6)

        # first block to start with input dim
        self.blocks = nn.ModuleList(
            [
                # start with the first block according to input dimension
                nn.ModuleList(
                    [
                        nn.Linear(2 * input_dim, dim1),
                        nn.Linear(dim1, dim1),
                        BatchNorm(dim1, momentum=momentum),
                        nn.Linear(dim1, dim1),
                        GravNetConv(
                            in_channels=dim1,
                            out_channels=dim1 * 2,
                            space_dimensions=space_dimensions,
                            k=k,
                            propagate_dimensions=dim1,
                        ),
                        BatchNorm(dim1 * 2, momentum=momentum),
                        nn.Linear(dim1 * 2, dim2),
                    ]
                )
            ]
        )

        # loop over remaining blocks as they are currently built the same
        self.blocks.extend(
            nn.ModuleList(
                [
                    # add according to number of blocks
                    nn.ModuleList(
                        [
                            nn.Linear(4 * dim1, dim1),
                            nn.Linear(dim1, dim1),
                            BatchNorm(dim1, momentum=momentum),
                            nn.Linear(dim1, dim1),
                            GravNetConv(
                                in_channels=dim1,
                                out_channels=dim1 * 2,
                                space_dimensions=space_dimensions,
                                k=k,
                                propagate_dimensions=dim1,
                            ),
                            # edges so need dim1 times 2
                            BatchNorm(dim1 * 2, momentum=momentum),
                            nn.Linear(dim1 * 2, dim2),
                        ]
                    )
                    for _ in range(nblocks - 1)
                ]
            )
        )

        # there are skip connections between the blocks,
        # this layer combines them, therefore scales with nblocks
        self.dense_cat = nn.Linear(dim2 * (nblocks), dim1)

        # These are the output layers for obj condensation

        self.p_beta_layer = nn.Linear(dim1, 1)  # predict condensation point
        self.p_ccoords_layer = nn.Linear(
            dim1, coord_dim
        )  # this gives latent space coordinates for clusters

        # These are the output layers for the track fitting
        self.p_p_layer = nn.Linear(dim1, 3)  # predict 3D momentum
        self.p_vertex_layer = nn.Linear(dim1, 3)  # predict track start point
        self.p_charge_layer = nn.Linear(
            dim1, 1
        )  # classification of positive or negative charge

    def forward(self, x, batch):
        # x, batch = data.x, data.batch

        feat = []
        x = self.batch_norm_0(x)
        # feat.append(x)

        # global exchange
        out = global_mean_pool(x, batch)
        x = torch.cat([x, out[batch]], dim=-1)

        # Apply Grav Net Blocks
        for i, block in enumerate(self.blocks):
            if i > 0:
                # do this for every block except input
                out = global_mean_pool(x, batch)
                x = torch.cat([x, out[batch]], dim=-1)
            x = F.elu(block[0](x))
            x = F.elu(block[1](x))
            # batch norm
            x = block[2](x)
            x = F.elu(block[3](x))
            # grav net
            x = block[4](x, batch)
            # batch norm 2
            x = block[5](x)
            # append output
            feat.append(F.elu(block[6](x)))  # skip connections

        # concat features and put through final dense NN
        x = torch.cat(feat, dim=1)
        x = F.elu(self.dense_cat(x))

        # Here are the networks for object condensation predictions
        p_beta = torch.sigmoid(self.p_beta_layer(x))
        p_ccoords = self.p_ccoords_layer(x)

        # Here are the networks for track fitting predictions
        p_p = self.p_p_layer(x)
        p_vertex = self.p_vertex_layer(x)
        p_charge = torch.sigmoid(self.p_charge_layer(x))

        # concatenate all predictions
        preds = torch.cat(
            (
                p_beta,
                p_ccoords,
                p_p,
                p_vertex,
                p_charge,
            ),
            dim=1,
        )
        return preds
