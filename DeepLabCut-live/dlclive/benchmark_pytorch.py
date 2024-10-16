import csv
import os
import platform
import subprocess
import sys
import time

import colorcet as cc
import cv2
import h5py
import numpy as np
import torch
from PIL import ImageColor
from pip._internal.operations import freeze

from dlclive import DLCLive
from dlclive.version import VERSION


def get_system_info() -> dict:
    """
    Returns a summary of system information relevant to running benchmarking.

    Returns
    -------
    dict
        A dictionary containing the following system information:
        - host_name (str): Name of the machine.
        - op_sys (str): Operating system.
        - python (str): Path to the Python executable, indicating the conda/virtual environment in use.
        - device_type (str): Type of device used ('GPU' or 'CPU').
        - device (list): List containing the name of the GPU or CPU brand.
        - freeze (list): List of installed Python packages with their versions.
        - python_version (str): Version of Python in use.
        - git_hash (str or None): If installed from git repository, hash of HEAD commit.
        - dlclive_version (str): Version of the DLCLive package.
    """

    # Get OS and host name
    op_sys = platform.platform()
    host_name = platform.node().replace(" ", "")

    # Get Python executable path
    if platform.system() == "Windows":
        host_python = sys.executable.split(os.path.sep)[-2]
    else:
        host_python = sys.executable.split(os.path.sep)[-3]

    # Try to get git hash if possible
    git_hash = None
    dlc_basedir = os.path.dirname(os.path.dirname(__file__))
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dlc_basedir)
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        # Not installed from git repo, e.g., pypi
        pass

    # Get device info (GPU or CPU)
    if torch.cuda.is_available():
        dev_type = "GPU"
        dev = [torch.cuda.get_device_name(torch.cuda.current_device())]
    else:
        from cpuinfo import get_cpu_info

        dev_type = "CPU"
        dev = [get_cpu_info()["brand_raw"]]

    return {
        "host_name": host_name,
        "op_sys": op_sys,
        "python": host_python,
        "device_type": dev_type,
        "device": dev,
        "freeze": list(freeze.freeze()),
        "python_version": sys.version,
        "git_hash": git_hash,
        "dlclive_version": VERSION,
    }


def analyze_video(
    video_path: str,
    model_path: str,
    model_type: str,
    device: str,
    precision: str = "FP32",
    snapshot: str = None,
    display=True,
    pcutoff=0.5,
    display_radius=5,
    resize=None,
    cropping=None,  # Adding cropping to the function parameters
    dynamic=(False, 0.5, 10),
    save_poses=False,
    save_dir="model_predictions",
    draw_keypoint_names=False,
    cmap="bmy",
    get_sys_info=True,
    save_video=False,
):
    """
    Analyzes a video to track keypoints using a DeepLabCut model, and optionally saves the keypoint data and the labeled video.

    Parameters
    ----------
    video_path : str
        Path to the video file to be analyzed.
    model_path : str
        Path to the DeepLabCut model.
    model_type : str
        Type of the model (e.g., 'onnx').
    device : str
        Device to run the model on ('cpu' or 'cuda').
    precision : str, optional, default='FP32'
        Precision type for the model ('FP32' or 'FP16').
    snapshot : str, optional
        Snapshot to use for the model, if using pytorch as model type.
    display : bool, optional, default=True
        Whether to display frame with labelled key points.
    pcutoff : float, optional, default=0.5
        Probability cutoff below which keypoints are not visualized.
    display_radius : int, optional, default=5
        Radius of circles drawn for keypoints on video frames.
    resize : tuple of int (width, height) or None, optional
        Resize dimensions for video frames. e.g. if resize = 0.5, the video will be processed in half the original size. If None, no resizing is applied.
    cropping : list of int or None, optional
        Cropping parameters [x1, x2, y1, y2] in pixels. If None, no cropping is applied.
    dynamic : tuple, optional, default=(False, 0.5, 10) (True/false), p cutoff, margin)
        Parameters for dynamic cropping. If the state is true, then dynamic cropping will be performed. That means that if an object is detected (i.e. any body part > detectiontreshold), then object boundaries are computed according to the smallest/largest x position and smallest/largest y position of all body parts. This window is expanded by the margin and from then on only the posture within this crop is analyzed (until the object is lost, i.e. <detection treshold). The current position is utilized for updating the crop window for the next frame (this is why the margin is important and should be set large enough given the movement of the animal).
    save_poses : bool, optional, default=False
        Whether to save the detected poses to CSV and HDF5 files.
    save_dir : str, optional, default='model_predictions'
        Directory to save output data and labeled video.
    draw_keypoint_names : bool, optional, default=False
        Whether to display keypoint names on video frames in the saved video.
    cmap : str, optional, default='bmy'
        Colormap from the colorcet library for keypoint visualization.
    get_sys_info : bool, optional, default=True
        Whether to print system information.
    save_video : bool, optional, default=False
        Whether to save the labeled video.

    Returns
    -------
    tuple
        A tuple containing:
        - poses (list of dict): List of pose data for each frame.
        - times (list of float): List of inference times for each frame.
    """

    # Create the DLCLive object with cropping
    dlc_live = DLCLive(
        path=model_path,
        model_type=model_type,
        device=device,
        display=display,
        resize=resize,
        cropping=cropping,  # Pass the cropping parameter
        dynamic=dynamic,
        precision=precision,
        snapshot=snapshot,
    )

    # Ensure save directory exists
    os.makedirs(name=save_dir, exist_ok=True)

    # Get the current date and time as a string
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Start empty dict to save poses to for each frame
    poses, times = [], []
    # Create variable indicate current frame. Later in the code +1 is added to frame_index
    frame_index = 0

    # Retrieve bodypart names and number of keypoints
    bodyparts = dlc_live.cfg["metadata"]["bodyparts"]
    num_keypoints = len(bodyparts)

    if save_video:
        # Set colors and convert to RGB
        cmap_colors = getattr(cc, cmap)
        colors = [
            ImageColor.getrgb(color)
            for color in cmap_colors[:: int(len(cmap_colors) / num_keypoints)]
        ]

        # Define output video path
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(
            save_dir, f"{video_name}_DLCLIVE_LABELLED_{timestamp}.mp4"
        )

        # Get video writer setup
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        vwriter = cv2.VideoWriter(
            filename=output_video_path,
            fourcc=fourcc,
            fps=fps,
            frameSize=(frame_width, frame_height),
        )

    while True:

        ret, frame = cap.read()
        if not ret:
            break
        # if frame_index == 0:
        #     pose = dlc_live.init_inference(frame)  # load DLC model
        try:
            # pose = dlc_live.get_pose(frame)
            if frame_index == 0:
                # dlc_live.dynamic = (False, dynamic[1], dynamic[2]) # TODO trying to fix issues with dynamic cropping jumping back and forth between dyanmic cropped and original image
                pose, inf_time = dlc_live.init_inference(frame)  # load DLC model
            else:
                # dlc_live.dynamic = dynamic
                pose, inf_time = dlc_live.get_pose(frame)
        except Exception as e:
            print(f"Error analyzing frame {frame_index}: {e}")
            continue

        poses.append({"frame": frame_index, "pose": pose})
        times.append(inf_time)

        if save_video:
            # Visualize keypoints
            this_pose = pose["poses"][0][0]
            for j in range(this_pose.shape[0]):
                if this_pose[j, 2] > pcutoff:
                    x, y = map(int, this_pose[j, :2])
                    cv2.circle(
                        frame,
                        center=(x, y),
                        radius=display_radius,
                        color=colors[j],
                        thickness=-1,
                    )

                    if draw_keypoint_names:
                        cv2.putText(
                            frame,
                            text=bodyparts[j],
                            org=(x + 10, y),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=colors[j],
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )

            vwriter.write(image=frame)
        frame_index += 1

    cap.release()
    if save_video:
        vwriter.release()

    if get_sys_info:
        print(get_system_info())

    if save_poses:
        save_poses_to_files(video_path, save_dir, bodyparts, poses, timestamp=timestamp)

    return poses, times


def save_poses_to_files(video_path, save_dir, bodyparts, poses, timestamp):
    """
    Saves the detected keypoint poses from the video to CSV and HDF5 files.

    Parameters
    ----------
    video_path : str
        Path to the analyzed video file.
    save_dir : str
        Directory where the pose data files will be saved.
    bodyparts : list of str
        List of body part names corresponding to the keypoints.
    poses : list of dict
        List of dictionaries containing frame numbers and corresponding pose data.

    Returns
    -------
    None
    """

    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    csv_save_path = os.path.join(save_dir, f"{base_filename}_poses_{timestamp}.csv")
    h5_save_path = os.path.join(save_dir, f"{base_filename}_poses_{timestamp}.h5")

    # Save to CSV
    with open(csv_save_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ["frame"] + [
            f"{bp}_{axis}" for bp in bodyparts for axis in ["x", "y", "confidence"]
        ]
        writer.writerow(header)
        for entry in poses:
            frame_num = entry["frame"]
            pose = entry["pose"]["poses"][0][0]
            row = [frame_num] + [
                item.item() if isinstance(item, torch.Tensor) else item
                for kp in pose
                for item in kp
            ]
            writer.writerow(row)

    # Save to HDF5
    with h5py.File(h5_save_path, "w") as hf:
        hf.create_dataset(name="frames", data=[entry["frame"] for entry in poses])
        for i, bp in enumerate(bodyparts):
            hf.create_dataset(
                name=f"{bp}_x",
                data=[
                    (
                        entry["pose"]["poses"][0][0][i, 0].item()
                        if isinstance(entry["pose"]["poses"][0][0][i, 0], torch.Tensor)
                        else entry["pose"]["poses"][0][0][i, 0]
                    )
                    for entry in poses
                ],
            )
            hf.create_dataset(
                name=f"{bp}_y",
                data=[
                    (
                        entry["pose"]["poses"][0][0][i, 1].item()
                        if isinstance(entry["pose"]["poses"][0][0][i, 1], torch.Tensor)
                        else entry["pose"]["poses"][0][0][i, 1]
                    )
                    for entry in poses
                ],
            )
            hf.create_dataset(
                name=f"{bp}_confidence",
                data=[
                    (
                        entry["pose"]["poses"][0][0][i, 2].item()
                        if isinstance(entry["pose"]["poses"][0][0][i, 2], torch.Tensor)
                        else entry["pose"]["poses"][0][0][i, 2]
                    )
                    for entry in poses
                ],
            )


import argparse
import os


def main():
    """Provides a command line interface to analyze_video function."""

    parser = argparse.ArgumentParser(
        description="Analyze a video using a DeepLabCut model and visualize keypoints."
    )
    parser.add_argument("model_path", type=str, help="Path to the model.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("model_type", type=str, help="Type of the model (e.g., 'DLC').")
    parser.add_argument(
        "device", type=str, help="Device to run the model on (e.g., 'cuda' or 'cpu')."
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="FP32",
        help="Model precision (e.g., 'FP32', 'FP16').",
    )
    parser.add_argument(
        "-s",
        "--snapshot",
        type=str,
        default=None,
        help="Path to a specific model snapshot.",
    )
    parser.add_argument(
        "-d", "--display", action="store_true", help="Display keypoints on the video."
    )
    parser.add_argument(
        "-c",
        "--pcutoff",
        type=float,
        default=0.5,
        help="Probability cutoff for keypoints visualization.",
    )
    parser.add_argument(
        "-dr",
        "--display-radius",
        type=int,
        default=5,
        help="Radius of keypoint circles in the display.",
    )
    parser.add_argument(
        "-r",
        "--resize",
        type=int,
        default=None,
        help="Resize video frames to [width, height].",
    )
    parser.add_argument(
        "-x",
        "--cropping",
        type=int,
        nargs=4,
        default=None,
        help="Cropping parameters [x1, x2, y1, y2].",
    )
    parser.add_argument(
        "-y",
        "--dynamic",
        type=float,
        nargs=3,
        default=[False, 0.5, 10],
        help="Dynamic cropping [flag, pcutoff, margin].",
    )
    parser.add_argument(
        "--save-poses", action="store_true", help="Save the keypoint poses to files."
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the output video with keypoints.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="model_predictions",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--draw-keypoint-names",
        action="store_true",
        help="Draw keypoint names on the video.",
    )
    parser.add_argument(
        "--cmap", type=str, default="bmy", help="Colormap for keypoints visualization."
    )
    parser.add_argument(
        "--no-sys-info",
        action="store_false",
        help="Do not print system info.",
        dest="get_sys_info",
    )

    args = parser.parse_args()

    # Call the analyze_video function with the parsed arguments
    analyze_video(
        video_path=args.video_path,
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        precision=args.precision,
        snapshot=args.snapshot,
        display=args.display,
        pcutoff=args.pcutoff,
        display_radius=args.display_radius,
        resize=tuple(args.resize) if args.resize else None,
        cropping=args.cropping,
        dynamic=tuple(args.dynamic),
        save_poses=args.save_poses,
        save_dir=args.save_dir,
        draw_keypoint_names=args.draw_keypoint_names,
        cmap=args.cmap,
        get_sys_info=args.get_sys_info,
        save_video=args.save_video,
    )


if __name__ == "__main__":
    main()
