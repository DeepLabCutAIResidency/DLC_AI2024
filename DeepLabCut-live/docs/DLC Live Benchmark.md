## Inference time

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

** **CUDA: NVIDIA GeForce RTX 3050 (6GB)**
** **CPU: 13th Gen Intel Core i7-13620H × 16** 
** **Linux: Ubuntu 24.04 LTS**

^ *Startup time at inference for a TensorRT engine takes between 30 and 50 seconds, which skews the inference time measurement. Caching is used to reduce that time.*

## Performance metrics

|     |     |
| --- | --- |
|     |     |
