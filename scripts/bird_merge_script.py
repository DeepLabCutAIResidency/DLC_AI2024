from deeplabcut.modelzoo.generalized_data_converter.datasets import (
    MaDLCPoseDataset,
    MultiSourceDataset,
    COCOPoseDataset,
    SingleDLCPoseDataset,
)


import deeplabcut
import os
import glob

all_bird_root = "/media/dikra/ADATA HD650/PhD/DATA/DLC24_Data/tiny_all_bird"

bird_proj_roots = glob.glob(os.path.join(all_bird_root, "*"))

datasets = {}

for proj_root in bird_proj_roots:

    if not os.path.isdir(proj_root):
        continue

    dataset_name = proj_root.split("/")[-1]

    if os.path.exists(os.path.join(proj_root, "config.yaml")):
        # a dlc project
        config_path = os.path.join(proj_root, "config.yaml")

        cfg = deeplabcut.auxiliaryfunctions.read_config(config_path)

        cfg["default_net_type"] = "top_down_resnet_50"

        deeplabcut.auxiliaryfunctions.write_config(config_path, cfg)

        if not os.path.exists(os.path.join(proj_root, "training-datasets")):
            print(f"creating training dataset for {proj_root}")
            deeplabcut.create_training_dataset(config_path)

        if "individuals" in cfg:

            datasets[dataset_name] = MaDLCPoseDataset(
                proj_root, dataset_name, shuffle=1
            )
        else:
            datasets[dataset_name] = SingleDLCPoseDataset(
                proj_root, dataset_name, shuffle=1
            )

    else:

        datasets[dataset_name] = COCOPoseDataset(proj_root, dataset_name)

    datasets[dataset_name].summary()

table_path = f"{all_bird_root}/conversion_table_bird.csv"
multi_dataset = MultiSourceDataset("superbird", list(datasets.values()), table_path)

multi_dataset.summary()

multi_dataset.materialize(
    "/media/dikra/ADATA HD650/PhD/DATA/DLC24_Data/tiny_all_bird_merged_madlc",
    framework="madlc",
    deepcopy=True,
)
