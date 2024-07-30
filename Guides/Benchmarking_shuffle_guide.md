# OUTLINE
## Reasoning for benchmarking models created across DLC 3.0 and earlier versions

DLC 3.0 runs on PyTorch as the engine rather than tensor flow (TF). It is of importance for replicability of data analysis to benchmark existing models created using DLC versions prior to 3.0 against new models created in DLC 3.0 and later versions.

The main element of importance for comparability when benchmarking different models is to keep the same test-train data split. If our models use differing train and test data sets, the models are not trained on the same data, and thus the reliability of model benchmarking is compromised. Creating a model using the same data split can be carried out both in GUI and command line, which this guide serves to outline the steps for.

## Important files & folders

```
dlc-project
|
|___dlc-models-pytorch
|   |__ pytorch_config.yaml
|   |__ pose_cfg.yaml
|
|___training-datasets
|   |__ metadata.yaml
|
|___config.yaml
```

## Benchmarking a TF model against a PyTorch model
### Creating a shuffle

Using the same DLC TensorFlow 'shuffle' to create a new DLC Pytorch 'shuffle'
- In GUI
    1. Front page > Load project > Open project folder > choose *config.yaml*
    2. Select *'Create training dataset'* tab
    3. Tick *Use an existing data split* option    
    ![create_from_existing](<GUI_benchmarking_TF_PT_guide.rtfd/Screenshot 2024-07-29 at 17.09.15.png>)
    4. Click 'View existing shuffles' to ...?
    ![view_existing_sh](<GUI_benchmarking_TF_PT_guide.rtfd/Screenshot 2024-07-29 at 17.10.29.png>)
    5. Choose the index of the training shuffle you want to replicate. Let's assume we want to replicate the train-test split from OpenfieldOct30-trainset95shuffle 3, in which split: 3 in this case, we insert in the From shuffle menu
    ![choose_existing_index](<GUI_benchmarking_TF_PT_guide.rtfd/Screenshot 2024-07-29 at 17.12.17.png>)
    6. In order to create this new dataset, set the shuffle option to an un-used shuffle (here 4)
    ![choose_new_index](<GUI_benchmarking_TF_PT_guide.rtfd/Screenshot 2024-07-29 at 17.36.44.png>)
    7. Check that the remaining attributes are specified like the original model you want to benchmark against - this information can be found in the model folder dlc-models-pytorch Iteration (here 0)  shuffle (in this case 3) train pytorch_config.yaml
    In this file, net_type will specify the network architecture (here a resnet_50)
    For weight initialisation, for any TF model, this is automatically set to TransferLearning - ImageNet.
    As augmentation methods implemented in the TF and PyTorch versions of DLC are not comparable,  this element does not have to be changed.
    ![other_attrb](<GUI_benchmarking_TF_PT_guide.rtfd/Screenshot 2024-07-29 at 17.41.09.png>)
    8. Click *'Create training dataset'* and move on to *'train network'*. Shuffle should be set to the new shuffle you entered at the previous step (in this case, 4)
    ![create_from_existing](<GUI_benchmarking_TF_PT_guide.rtfd/Screenshot 2024-07-29 at 17.47.10.png>)
    9. If you wish to keep the training attributes identical  file to your initial TF model, this information can likewise be found in the pytorch_config.yaml
    runner:
	snapshots:	
		max_snapshots: = number of snapshots to keep
		save_epochs: = save epochs
    Train_settings:
        display_iters: Display iterations
    	epochs:  = Save epochs

- In Code 

    With the *deeplabcut* module in Python, use the *create_training_dataset_from_existing_split()* method to create new shuffles from existing ones (e.g. TensorFlow shuffles)

    Similarly, here we create a new shuffle '4' from the existing shuffle '3'.

    ```python
    import deeplabcut

    config = "path/to/project/config.yaml"

    training_dataset = deeplabcut.create_training_dataset_from_existing_split(config=config, from_shuffle=3, from_trainsetindex=0, shuffles=[4], net_type="resnet_50")
    ```
    We can then move to training our new PyTorch model with the same data split as the TensorFlow model.
    ```python
    deeplabcut.train_network(config, shuffle=4, engine=Engine.PYTORCH, batch_size=8)
    ```

    Once, trained we can evaluate our model using
    
    ```python
    deeplabcut.evaluate_network(config, Shuffles=[4], snapshotindex="all")
    ```
    Now, we are able to compare performances with peace of mind!
