# DLC Benchmarking - User Guide

## Reasoning for benchmarking models in DLC (across DLC versions and architectures)

DLC 3.0 runs on PyTorch as the engine rather than TensorFlow. It is of importance for
replicability of data analysis to benchmark existing models created using DLC versions
prior to 3.0 against new models created in DLC 3.0 and later versions.

When comparing different models, it's important to use the same train-test data split 
to ensure fair comparisons. If the models are trained on different datasets, 
their performance metrics can't be accurately compared. This is crucial when comparing 
the performance of models with different architectures or different sets of hyperparameters. 
For example, if we compare the RMSE of a model on an "easy" test image with the RMSE 
of another model on a "hard" test image, it doesn't determine whether a model is better 
than the other because the architecture performs better or because the training images 
were "better" to learn from. Thus, we not only need to compare the models based on metrics
computed on the same test images, but also train them on an identical fixed training set
in order to "decouple" the dataset from the model architecture

Creating a model using the same data split can be carried out both in GUI and command
line, which this guide serves to outline the steps for.

## Important files & folders

```
dlc-project
|
|___dlc-models-pytorch
|   |__ iterationX
|       |__ shuffleX
|           |__ pytorch_config.yaml
|  
|___training-datasets
|   |__ metadata.yaml
|
|___config.yaml
```

## Benchmarking a TensorFlow model against a PyTorch model

### Creating a shuffle

Creating a new shuffle with the same train/test split as an existing one:
#### In the DLC GUI
1. Front page > Load project > Open project folder > choose *config.yaml*
2. Select *'Create training dataset'* tab
3. Tick *Use an existing data split* option    

    ![create_from_existing](<assets/Screenshot 2024-07-29 at 17.09.15.png>)
4. Click 'View existing shuffles':
    - This is used to view the indices of shuffles that have been created for a project,
in order to determine which index is available to assign to a new shuffle.
    - The elements described in this window are:
        - train_fraction: The fraction of the dataset used for training.
        - index: The index of the shuffle.
        - split: The data split for the shuffle. The integer value on its own does not
hold any meaning, but this "split" value indicates which shuffles have the same split 
(as their results can then be compared)
        - engine: Whether it's a PyTorch or TensorFlow shuffle

            ![view_existing_sh](<assets/Screenshot 2024-07-29 at 17.10.29.png>)
5. Choose the index of the training shuffle you want to replicate. Let's assume we want
to replicate the train-test split from OpenfieldOct30-trainset95shuffle3, in which
`split: 3`. In this case, we insert in the *'From shuffle'* menu
    
    ![choose_existing_index](<assets/Screenshot 2024-07-29 at 17.12.17.png>)
6. In order to create this new dataset, set the shuffle option to an un-used shuffle
(here 4)
    
    ![choose_new_index](<assets/Screenshot 2024-07-29 at 17.36.44.png>)
7. Click *'Create training dataset'* and move on to *'train network'*. Shuffle should be 
set to the new shuffle you entered at the previous step (in this case, 4)
    
    ![create_from_existing](<assets/Screenshot 2024-07-29 at 17.47.10.png>)
8. If you wish to keep the training attributes identical to your initial TensorFlow
model, specifications of the original model can be found in the model folder
dlc-models-pytorch > iteration folder (here 0) > shuffle (in this case 3) > train > 
pytorch_config.yaml. Here all parameters of the original model can be found.

#### In Code 

With the `deeplabcut` module in Python, use the
`create_training_dataset_from_existing_split()` method to create new shuffles from
existing ones (e.g. TensorFlow shuffles).

Similarly, here we create a new shuffle '4' from the existing shuffle '3'.

```python
import deeplabcut
from deeplabcut.core.engine import Engine

config = "path/to/project/config.yaml"

training_dataset = deeplabcut.create_training_dataset_from_existing_split(
   config=config,
   from_shuffle=3,
   from_trainsetindex=0,
   shuffles=[4],
   net_type="resnet_50",
)
```

We can then move to training our new PyTorch model with the same data split as the
TensorFlow model.

```python
deeplabcut.train_network(config, shuffle=4, engine=Engine.PYTORCH, batch_size=8)
```

Once, trained we can evaluate our model using

```python
deeplabcut.evaluate_network(config, Shuffles=[4], snapshotindex="all")
```
Now, we are able to compare performances with peace of mind!

#### Good practice: naming shuffles created from existing ones

In a setting where one has multiple TensorFlow models, and intends to benchmark their
performances again new PyTorch models, it is good practice to follow a naming pattern
for the shuffles we create.

In practice, let's say we have TensorFlow shuffles 0, 1, 2, we can create new PyTorch
shuffles from them by naming them: 1000, 1001, 1002. This allows us to quickly recognise
that the shuffles belonging to the 100x range are PyTorch shuffles and that shuffle
1001, for example, has the same data split at the TensorFlow shuffle 1. This way, the
comparison can be more straightforward and guaranteed to be correct!
