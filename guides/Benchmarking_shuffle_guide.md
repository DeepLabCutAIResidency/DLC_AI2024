# OUTLINE
## Reasoning for benchmarking models created across DLC 3.0 and earlier versions

DLC 3.0 runs on PyTorch as the engine rather than TensorFlow (TF). It is of importance for replicability of data analysis to benchmark existing models created using DLC versions prior to 3.0 against new models created in DLC 3.0 and later versions.

When benchmarking different models, maintaining the same test-train data split is crucial for ensuring comparability. If the models use differing train and test datasets, their performance metrics cannot be accurately compared because they are not trained on the same data. This is especially important when comparing the performance of different models, such as a TensorFlow model and a PyTorch model, or two models with different architectures. Using the same training set is necessary to ensure fair comparisons, as different training sets may yield different results and make it difficult to accurately compare the models' performance.

## Important files & folders

```
dlc-project
|
|___dlc-models-pytorch
|   |__ pytorch_config.yaml
|
|___training-datasets
|   |__ metadata.yaml
|
|___config.yaml
```

## Benchmarking a TF model against a PyTorch model
### Creating a shuffle

Creating a new shuffle with the same train/test split as an existing one
#### In the DLC GUI
1. Front page > Load project > Open project folder > choose *config.yaml*
2. Select *'Create training dataset'* tab
3. Tick *Use an existing data split* option    
![create_from_existing](<assets/Screenshot 2024-07-29 at 17.09.15.png>)
4. Click 'View existing shuffles':
    - This can be used to view the indices of shuffles that have been created for a project (otherwise you can't know which index is available to be given to the new shuffle).
    - The elements described in this window are:
        - train_fraction: The fraction of the dataset used for training.
        - index: The index of the shuffle.
        - split: The data split for the shuffle. There's no need to "interpret" this value as-is (as the integer value doesn't mean much). What matters with this "split" value if which shuffles have the same split (as their test results can then be compared)
        - engine: Whether it's a PyTorch or TensorFlow shuffle
![view_existing_sh](<assets/Screenshot 2024-07-29 at 17.10.29.png>)
5. Choose the index of the training shuffle you want to replicate. Let's assume we want to replicate the train-test split from OpenfieldOct30-trainset95shuffle 3, in which split: 3 in this case, we insert in the From shuffle menu
![choose_existing_index](<assets/Screenshot 2024-07-29 at 17.12.17.png>)
6. In order to create this new dataset, set the shuffle option to an un-used shuffle (here 4)
![choose_new_index](<assets/Screenshot 2024-07-29 at 17.36.44.png>)
7. Click *'Create training dataset'* and move on to *'train network'*. Shuffle should be set to the new shuffle you entered at the previous step (in this case, 4)
![create_from_existing](<assets/Screenshot 2024-07-29 at 17.47.10.png>)
8. If you wish to keep the training attributes identical  file to your initial TF model, this information can likewise be found in the pytorch_config.yaml
runner:
snapshots:	
    max_snapshots: = number of snapshots to keep
    save_epochs: = save epochs
Train_settings:
    display_iters: Display iterations
    epochs:  = Save epochs

#### In Code 

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

#### Good practice: naming shuffles created from existing ones
In a setting where one has multiple TF models, and intends to benchmark their performances again new PyTorch models, it is good practice to follow a naming pattern for the shuffles we create.

In practice, let's say we have TensorFlow shuffles 0, 1, 2, we can create new PyTorch shuffles from them by naming them: 1000, 1001, 1002. This allows us to quickly recognise that the shuffles belonging to the 100x range are PyTorch shuffles and that shuffle 1001, for example, has the same data split at the TensorFlow shuffle 1. This way, the comparison can be more straighforward and guaranteed to be correct!

