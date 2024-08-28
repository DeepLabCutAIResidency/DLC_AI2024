This repository contains a [DeepLabCut](http://www.mousemotorlab.org/deeplabcut) inference pipeline for real-time applications that has minimal (software) dependencies. This new DLC Live pipeline can handle DLC models produced in PyTorch, as of DLC 3.0.

In DLC Live TensorFlow, model export is handled in the main DLC package. 
The current pipeline handles both model export and video inference. The means that currently DLCLive still uses the main DLC package as well. The aim is for the model export to be a functio of the main DLC package, allowing DLC Live to be a standalone package with minimal software dependencies.

**Contents of this package:** This package provides a `DLCLive` class which enables pose estimation online to provide feedback. This object loads and prepares a DeepLabCut network for inference, and will return the predicted pose for single images. 

In the future this package should also contain a `Processor` object. We have not yet had time to work on this object, and the code currently provides a placeholder object for `Processor` that does not carry out any specific function.

For details on the `Processor` object in DLCLive TensorFlow, see documentation [here](dlclive/processor/README.md).

###### 🎥🎥🎥 Note :: alone, this object does not record video or capture images from a camera. This pipeline provides scripts to run video inference on a prerecorded video, as well as video inference on a live video feed.🎥🎥🎥


### Quick Start: instructions for use:

To use DLCLive 3.0, two methods for doing so are provided. Besides using the provided scripts to open a camera feed (live or prerecorded), it is also possible to use the DLCLive object directly. 
TODO: does the current version of DLCLive support this?
1. Initialize `Processor` (if desired) - the current version of this repo has not actively worked with different processors.
2. Initialize the `DLCLive` object
3. Perform pose estimation!

```python
from dlclive import DLCLive, Processor
dlc_proc = Processor()
dlc_live = DLCLive(<path to exported model directory>, processor=dlc_proc)
dlc_live.init_inference(<your image>)
dlc_live.get_pose(<your image>)
```

`DLCLive` **inputs:**
  - `<your image>` = is a numpy array of each frame
  - `<path to exported model directory>` = path to the folder that has the `.pt` and config file (for pytorch projects) or `.onnx` files. The .onnx file is acquired after using the torch.onnx.export function (see demo notebook for examples of how to do so). `.pt` and config files are provided in the folder of the DLC project.  

  ```
dlc-project
|
|___dlc-models-pytorch
|   |__ iterationX
|       |__ shuffleX
|           |__pytorch_config.yaml
|           |__snapshot-X.pt
|  

```
  
  TODO: add docs instructions for model export using ONNX.



Both video inference on pre-recorded and live video feeds use the `DLCLive` as the central component.

`DLCLive` **parameters:**

  - `path` = string; full path to the exported DLC model directory
  - `device` = str, default = cpu; whether the model should run on GPU (if available) or CPU.
  - `model_type` = string; the type of model to use for inference. Default = onnx. Types include:
      - `pytorch` = the base DeepLabCut model produced by DLC 3.0 using PyTorch as the engine 
      - `onnx` = this assumes the user has exported their PyTorch model (using a .pt snapshot and pytorch_config file) to an ONNX model [onnx](https://onnxruntime.ai/pytorch) and uses ONNX runtime
      - `tensorrt` = currently compatible with ONNX models but not PyTorch models, and uses a TensorRT runtime ##DIKRA is this correctly specified?
  - `precision` = string, optional
        precision of model weights. Can be 'FP32' (default) or 'FP16'.
  - `cropping` = list of int, optional; cropping parameters in pixel number: [x1, x2, y1, y2]
  - `dynamic` = triple, containing (state, detectiontreshold, margin). If not wanting to use dynamic cropping, state should be set to False. default = (False, 0.5, 10). 
      - If the state is true, then dynamic cropping will be performed. That means that if an object is detected (i.e. any body part > detectiontreshold), then object boundaries are computed according to the smallest/largest x position and smallest/largest y position of all body parts. This  window is expanded by the margin and from then on only the posture within this crop is analyzed (until the object is lost, i.e. <detectiontreshold). The current position is utilized for updating the crop window for the next frame (this is why the margin is important and should be set large enough given the movement of the animal).
  - `resize` = float, optional; factor by which to resize image (resize=0.5 downsizes both width and height of image by half). Can be used to downsize large images for faster inference.
  - `processor` = dlc pose processor object, optional. The current code has not yet done any active use of the processor parameter.
  - `display` = bool, optional; display processed image with DeepLabCut points. Can be used to troubleshoot cropping and resizing parameters, but can be slow
  - `pcutoff` = float, default = 0.5; confidence cut-off for displaying key points.
  - `convert2rgb` = bool, optional; boolean flag to convert frames from BGR to RGB color scheme
  - `display_radius` = int, default = 3; radius of the points on the video display
  - `display_cmap`= str, default = "bmy"; color scheme of the key points on the display


#### Option 1: Video inference on pre-recorded videos

TODO add __main__ element to script to use it in bash as well, not solely for iporting functions.

TODO add function for making saving a video optional

The benchmark_pytorch.py script provides the 'analyze_video' function for doing video inference and benchmark inference speed on a pre-recorded video.

The function takes the same parameters as `DLCLive` as it directly calls the DLCLive object. In addition, it takes the following arguments:

  - `video_path` = string; The path to the video file to be analyzed.
  - `save_poses` = bool, optional, default=False; Whether to save the detected poses to CSV and HDF5 files.
  - `save_dir` = str, optional, default="model_predictions"; The directory where the output video and pose data will be saved.
  - `draw_keypoint_names` = draw_keypoint_names : bool, optional, default=False; Whether to draw the names of the keypoints on the video frames.
  - `cmap` = str, optional, default="bmy"; The colormap from the colorcet library to use for keypoint visualization.
  - `get_sys_info` = bool, optional, default=True; Whether or not to obtain the information of the system running the video inference. Will be printed, not saved. TODO save this information

Example of how to run the function:
##### python
```python
from dlclive.benchmark_pytorch import analyze_video

analyze_video(model_path='/path/to/exported/model', video_path='/path/to/video', save_dir='/path/to/output', resize=0.5)
```


#### Option 2: Inference on live video feed


TODO add __main__ element to script to use it in bash as well, not solely for iporting functions.


TODO add function for making saving a video optional

The LiveVideoInference.py script provides the 'analyze_live_video' function for doing live video tracking.

The function is similar in function to 'analyze_video' but replaces `video_path` with the following arguments:

  - `camera` = float, default = 0; The camera to record the live video from. The default '0' is the webcam if using a laptop. If using a USB webcam, this will typically obtain the number of one of the USB ports, e.g. 1. If using a camera such a Basler, this will likely need confugiration. We have not tested this.
  - `experiment_name` = str, default = "Test"; Prefix to label generated pose and video files.

Example of how to run the function:
##### python
```python
from dlclive.LiveVideoInference import analyze_live_video

analyze_video(model_path='/path/to/exported/model', camera=0, experiment_name = "experiment_20240827", save_dir='/path/to/output', resize=0.5)
```
**Demo Notebook:**
this repository contains a notebook for demonstrating the functionalities of the DLCLive 3.0. This can be found in [DLCLive-Demo.ipynb](https://github.com/DeepLabCutAIResidency/DLC_AI2024/blob/main/DeepLabCut-live/DLCLive-Demo.ipynb)

**Benchmarking dataset and model:** Estimates of performance is carried out using a PyTorch ResNet50 model and an associated video recording (INSERT LINK).
Code is in progress for running DLC Live using other models (currenly focusing on hr_net)

**Performance:** Benchmarking results are available below.

| System | Model type | Runtime  | Device type | Precision                              | Video        | Video length (s) - # Frames | FPS | Frame size | Pose model backbone | Avg Inference time ± Std <br>*(including 1st inference)* | Avg Inference time ± Std | Average FPS ± Std | Model size |
| ------ | ---------- | -------- | ----------- | -------------------------------------- | ------------ | --------------------------- | --- | ---------- | ------------------- | -------------------------------------------------------- | ------------------------ | ----------------- | ---------- |
| Linux  | ONNX       | ONNX     | CUDA        | Full precision (FP32)                  | Ventral gait | 10s - 1.5k                  | 150 | (658,302)  | `ResNet50` (bu)     | 29.02ms ± 47.59ms                                        | 27.8ms ± 2.32ms          | 36 ± 3            | 92.12 MB   |
| Linux  | ONNX       | ONNX     | CPU         | Full precision (FP32)                  | Ventral gait | 10s - 1.5k                  | 150 | (658,302)  | `ResNet50` (bu)     | 146.12ms ± 13.26ms                                       | 146.11 ± 13.25           | 7 ± 1             | 92.12 MB   |
| Linux  | PyTorch    | PyTorch  | CUDA        | Full precision (FP32)                  | Ventral gait | 10s - 1.5k                  | 150 | (658,302)  | `ResNet50` (bu)     | 6.04ms ± 7.37ms                                          | 5.97ms ± 6.8ms           | 271 ± 112         | 96.5 MB    |
| Linux  | PyTorch    | PyTorch  | CPU         | Full precision (FP32)                  | Ventral gait | 10s - 1.5k                  | 150 | (658,302)  | `ResNet50` (bu)     | 365.26ms ± 13.88ms                                       | 365.17ms ± 13.44ms       | 3 ± 0             | 96.5 MB    |
| Linux  | ONNX       | TensorRT | CUDA        | Full precision (FP32) - no caching     | Ventral gait | 10s - 1.5k                  | 150 | (658,302)  | `ResNet50` (bu)     | 55.32ms ± 1254.16ms^                                     | 22.93ms ± 0.88           | 44 ± 2            | 92.12 MB   |
| Linux  | ONNX       | TensorRT | CUDA        | Full precision (FP32) - engine caching | Ventral gait | 10s - 1.5k                  | 150 | (658,302)  | `ResNet50` (bu)     | 20.8ms ± 3.4ms                                           | 20.72ms ± 1.25ms         | 48 ± 3            | 92.12 MB   |
| Linux  | ONNX       | TensorRT | CUDA        | FP16                                   | Ventral gait | 10s - 1.5k                  | 150 | (658,302)  | `ResNet50` (bu)     | 34.37ms ± 858.96ms                                       | 12.19ms ± 0.87           | 82 ± 6            | 46.16 MB   |
| Linux  | ONNX       | ONNX     | CUDA        | FP16                                   | Ventral gait | 10s - 1.5k                  | 150 | (658,302)  | `ResNet50` (bu)     | 21.74ms ± 43.24ms                                        | 20.62ms ± 2.5ms          | 49 ± 5            | 46.16 MB   |