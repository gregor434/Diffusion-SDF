#!/usr/bin/env python3

import os
import torch
import torch.utils.data

import numpy as np

from dataloader.conditioning import build_conditioning_sources

class ModulationLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, split_file=None, conditioning=None):
        super().__init__()

        self.conditioning_sources = build_conditioning_sources(conditioning)
        self.conditional = len(self.conditioning_sources) > 0

        self.records = self.load_records(data_path, split_file)
        self.modulations = [
            torch.from_numpy(np.loadtxt(record["latent_path"])).float()
            for record in self.records
        ]
        #self.modulations = self.modulations[0:8]
        #pc_paths = pc_paths[0:8]

        print("data shape, dataset len: ", self.modulations[0].shape, len(self.modulations))
        #assert args.batch_size <= len(self.modulations)

    def __len__(self):
        return len(self.modulations)

    def __getitem__(self, index):

        record = self.records[index]
        conditioning = {
            source.name: source.load(record)
            for source in self.conditioning_sources
        }
        pc = conditioning.get("point_cloud", False)

        return {
            "point_cloud" : pc,
            "conditioning": conditioning,
            "latent" : self.modulations[index]         
        }

    def load_records(self, data_source, split, f_name="latent.txt"):
        records = []
        for dataset in split: # dataset = "acronym"
            for class_name in split[dataset]:
                for instance_name in split[dataset][class_name]:
                    instance_filename = os.path.join(data_source, class_name, instance_name, f_name)
                    if not os.path.isfile(instance_filename):
                        continue

                    record = {
                        "dataset": dataset,
                        "class_name": class_name,
                        "instance_name": instance_name,
                        "latent_path": instance_filename,
                    }
                    if not all(source.exists(record) for source in self.conditioning_sources):
                        continue
                    records.append(record)
        return records
