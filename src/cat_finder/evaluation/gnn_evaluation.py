from cat_finder.evaluation import inference
import torch

torch.set_num_threads(4)
from torch_geometric.loader import DataLoader
import numpy as np

# from objcond.nn_training.modules import inference

from cat_finder.training import gnn_model, datatools, utils
import pandas as pd
import yaml
from collections import Counter

DEBUG = False


class Evaluation:
    def __init__(
        self,
        outputdir="./",
        model="best_model",
        config="config.yaml",
        device="cpu",
    ):
        # specify which run and model to run
        self.outputdir = outputdir
        self.model = model
        self.device = device

        # get respective config used for training
        self.cfg_path = f"{self.outputdir}/{config}"
        with open(self.cfg_path) as config_file:
            self.config = yaml.safe_load(config_file)

        self.net = (
            gnn_model.CDCNet(
                input_dim=len(self.config["dataset"]["input_features"]),
                k=self.config["model"]["k"],
                nblocks=self.config["model"]["blocks"],
                coord_dim=self.config["model"]["coord_dim"],
                dim1=self.config["model"]["dim1"],
                dim2=self.config["model"]["dim2"],
                space_dimensions=self.config["model"]["space_dimensions"],
                momentum=self.config["model"]["momentum"],
            )
            .float()
            .to(device)
        )
        # load model

        loaded_model = utils.load_model(modelpath=self.outputdir + f"{self.model}.pt")

        print(f'Best model from epoch:{loaded_model["epoch"]} ')
        self.net.load_state_dict(loaded_model["model_state_dict"])
        # set model to evaluation
        self.net.eval()

    def get_dataset(
        self,
        evt_type=None,
        n_events=None,
        input_dir=None,
    ):

        self.gd = datatools.CDCDataset(
            features=self.config["dataset"]["input_features"],
            n_events=(
                n_events if n_events else self.config["dataset"]["samples_per_file"]
            ),
            evtname=evt_type if evt_type else self.config["dataset"]["evt_type"],
            sampledir=(input_dir if input_dir else self.config["dataset"]["input_dir"]),
            truth=self.config["dataset"]["truth"],
            scaling=self.config["dataset"]["scaling"],
            clipping=self.config["dataset"]["clipping"],
        )

    def evaluate_minimal_dataset(
        self,
        test_start=0,
        test_size=1000,
        batch_size=32,
        radius=0.3,
        hit_radius=0.15,
        threshold=0.3,
        min_hit_num=7,
    ):
        self.test_size = test_size
        self.prediction_targets = ["px", "py", "pz", "vx", "vy", "vz", "charge"]

        df = pd.DataFrame()

        test_data = DataLoader(
            self.gd[test_start : test_start + test_size],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        with torch.no_grad():
            for j, test_set in enumerate(test_data):
                pred = self.net(
                    test_set.x.to(self.device), test_set.batch.to(self.device)
                )

                inf = inference.InfDicts(test_set.to(self.device), pred.to(self.device))

                inf_dict = inf.get_infdict(
                    coord_dim=self.config["model"]["coord_dim"],
                )
                inf_dict = inf_dict
                beta_mask = inf.get_betamask(threshold, radius).numpy()

                # loop over batch size
                for i in range(inf_dict["t_mask"].shape[0]):
                    tevent = test_start + i + batch_size * j
                    truth = inf_dict["truth"][i]
                    bmask = beta_mask[i]
                    prediction = inf_dict["p_prediction"][i]
                    prediction_obj = prediction[bmask]
                    con_point_coords = inf_dict["p_ccoords"][i][bmask]

                    ptracks = np.sum(bmask)

                    for b, obj in enumerate(prediction_obj):
                        (
                            matched_obj,
                            purity,
                            efficiency,
                            matched_hit_num,
                            true_hit_num,
                        ) = get_hit_purity_and_eff(
                            con_point_coords[b], inf_dict, hit_radius, i
                        )
                        if matched_hit_num > min_hit_num:
                            tmp_dict = {
                                "event": tevent,
                                "matched_obj": matched_obj,
                                "purity": purity,
                                "efficiency": efficiency,
                                "matched_hit_num": matched_hit_num,
                                "true_hit_num": true_hit_num,
                                "track": b,
                            }
                            # add predicted values
                            for pfidx, predicted_feature in enumerate(
                                self.prediction_targets
                            ):
                                tmp_dict[predicted_feature] = obj[pfidx].numpy()
                            tmp_df = pd.DataFrame(tmp_dict, index=[0])
                            df = pd.concat([df, tmp_df], ignore_index=True)
        self.df = df

    def save_results(self, outputdir=None, filename="evaluation_results"):
        if outputdir is None:
            outputdir = self.outputdir
        self.df.to_csv(f"{outputdir}/{filename}.csv")


def get_hit_purity_and_eff(con_point_coord, inf_dict, radius, i):
    coord = inf_dict["p_ccoords"][i, :, :]

    r = torch.sqrt(torch.sum((con_point_coord - coord) ** 2, dim=1))
    mask = r < radius

    hits = inf_dict["t_objidx"][i].numpy()
    assigned_hits = hits[mask.numpy()]
    assigned_counter = Counter(assigned_hits)
    hit_counter = Counter(hits)
    # what is the most common objid and how often does it occur
    most_common_num, num_occurrences = assigned_counter.most_common(1)[0]
    # calculate purity for this object
    purity = num_occurrences / len(assigned_hits)

    # how many hits belong to the truth
    hit_occurences = hit_counter[most_common_num]
    # calculate efficiency for this
    efficiency = num_occurrences / hit_occurences
    return most_common_num, purity, efficiency, num_occurrences, hit_occurences
