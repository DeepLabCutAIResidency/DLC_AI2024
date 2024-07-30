# OUTLINE
- Why?
-   DLC 3.0 comes when we already have models
-   we need to be able to benchmark our own models for our data to be comparable, at least theoretically.
-   why convert from TF to pytorch?
- how?
-   

- How to - GUI

- How to - code

# Why ?
(relevant use cases, why going from tf to pytorch ???)
# How to ?
(in case one has already trained a TensorFlow model and wants to do compare it to a Pytorch one)
## 1. Creating a shuffle
(Highlight why it is important to use the same shuffle to compare performances) 

Using the same DLC TensorFlow 'shuffle' to create a new DLC Pytorch 'shuffle'
- In GUI

    screenshot of 'Create from existing data split button'

    hilghlight the 'View existing shuffles button'
- In Code 

    > *deeplabcut.create_training_dataset_from_existing_split(config, from_shuffle, from_trainsetindex, shuffles, net_type)*

### 1.1 Important files & folders
(specify complete paths, and add tree diagrams)
```
dlc-project
|
|___dlc-models-pytorch
    |__ pytorch_config.yaml
    |__ pose_cfg.yaml

training-datasets
|__ metadata.yaml

config.yaml
```