import torch
from torch_geometric.data import Data, InMemoryDataset

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import os
import random

DEBUG = False


class CDCDataset(InMemoryDataset):
    def __init__(
        self,
        evtname,
        n_batches=None,
        features=["tdc", "adc", "x", "y"],
        truth=[
            "is_signal",
            "objidx",
            "cdchit_primary_px",
            "cdchit_primary_py",
            "cdchit_primary_pz",
        ],
        n_events=2000,
        transform=None,
        sampledir="./",
        torch_dir=None,
        pre_transform=None,
        clipping=None,
        seed=None,
        scaling=None,
    ):
        self.n_events = n_events
        self.eventname = evtname
        self.n_batches = n_batches
        self.features = features
        self.seed = seed
        self.sampledir = sampledir
        if torch_dir == None:
            torch_dir = f"{sampledir}/pytorch/"

        self.torch_dir = torch_dir
        self.scaling = scaling
        self.clipping = clipping
        if truth is None:
            self.truth = [
                "is_signal",
                "objidx",
                "cdchit_primary_px",
                "cdchit_primary_py",
                "cdchit_primary_pz",
            ]
        else:
            self.truth = truth

        super().__init__(transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.sampledir

    @property
    def processed_dir(self):
        total_dir = self.torch_dir
        os.makedirs(total_dir, exist_ok=True)
        return total_dir

    @property
    def raw_file_names(self):
        files = [f for f in os.listdir(self.sampledir) if f.endswith(".csv")]
        print("FILES", files)
        return files

    @property
    def processed_file_names(self):
        return f"{self.eventname}_{int(self.n_events)}_{self.n_batches}.pt"

    def download(self):
        pass

    def process(self):
        data_list = []
        print("Start Processing")
        for f in self.raw_file_names:
            filename = os.path.join(self.raw_dir, f)
            print(filename)
            if os.path.exists(filename):
                hit_df = pd.read_csv(filename)
                """
                update df info here
                """
                if self.clipping != None:
                    for variable in self.clipping:
                        hit_df[variable] = hit_df[variable].clip(
                            self.clipping[variable][0], self.clipping[variable][1]
                        )

                if self.scaling != None:
                    for variable in self.scaling:
                        # print(variable, self.scaling[variable])
                        hit_df[variable + "_unscaled"] = hit_df[variable]
                        frac = self.scaling[variable][0]
                        shift = self.scaling[variable][1]
                        hit_df[variable] = (hit_df[variable] - shift) / frac

                hit_df["positive_charge"] = hit_df["cdchit_mc_charge"] > 0

                # check how many events are in the file, and if its less than
                # the specified number, take max number of events in file instead
                events_list = hit_df["evt_id"].unique()
                events_in_file = len(events_list)
                nevents = (
                    self.n_events if events_in_file > self.n_events else events_in_file
                )
                print(
                    f"Loading samples from {filename}, {events_in_file} available in file, selecting {nevents}"
                )
                # loop over the events to build the graphs
                for eventid in range(0, nevents):
                    event_id = events_list[eventid]
                    hit_event_df = hit_df.query(f"evt_id == {event_id}")

                    feats_tensor = torch.tensor(
                        np.array(hit_event_df[self.features], dtype="float")
                    ).float()

                    truth_tensor = torch.tensor(
                        np.array(hit_event_df[self.truth], dtype="float")
                    ).float()

                    # create data
                    graph = Data(x=feats_tensor, y=truth_tensor)

                    data_list.append(graph)
        # shuffle list here to switch between ntracks,
        random.seed(self.seed)
        random.shuffle(data_list)
        data, slices = self.collate(data_list)
        torch.save(
            (data, slices), os.path.join(self.processed_dir, self.processed_file_names)
        )
