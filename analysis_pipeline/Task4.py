import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

COLORS = [
    "back",
    "belly",
    "bill",
    "breast",
    "crown",
    "forehead",
    "left_eye",
    "left_leg",
    "left_wing_tip",
    "left_wrist",
    "nape",
    "right_eye",
    "right_leg",
    "right_wing_tip",
    "right_wrist",
    "tail_tip",
    "throat",
    "neck",
    "tail_left",
    "tail_right",
    "upper_spine",
    "upper_half_spine",
    "lower_half_spine",
    "right_foot",
    "left_foot",
    "left_half_chest",
    "right_half_chest",
    "chin",
    "left_tibia",
    "right_tibia",
    "lower_spine",
    "upper_half_neck",
    "lower_half_neck",
    "left_chest",
    "right_chest",
    "upper_neck",
    "left_wing_shoulder",
    "left_wing_elbow",
    "right_wing_shoulder",
    "right_wing_elbow",
    "upper_cere",
    "lower_cere",
]

COLOR_MAP = {
    "back": (128, 0, 128),
    "bill": (0, 102, 204),
    "belly": (255, 128, 0),
    "breast": (0, 255, 255),
    "crown": (0, 255, 0),
    "forehead": (255, 105, 180),
    "left_eye": (102, 0, 204),
    "left_leg": (139, 69, 19),
    "left_wing_tip": (75, 0, 130),
    "left_wrist": (255, 140, 0),
    "nape": (255, 51, 51),
    "right_eye": (255, 255, 102),
    "right_leg": (205, 133, 63),
    "right_wing_tip": (30, 144, 255),
    "right_wrist": (50, 205, 50),
    "tail_tip": (0, 255, 127),
    "throat": (255, 20, 147),
    "neck": (0, 191, 255),
    "tail_left": (218, 112, 214),
    "tail_right": (255, 165, 0),
    "upper_spine": (32, 178, 170),
    "upper_half_spine": (0, 128, 128),
    "lower_half_spine": (135, 206, 235),
    "right_foot": (255, 69, 0),
    "left_foot": (128, 128, 0),
    "left_half_chest": (233, 150, 122),
    "right_half_chest": (220, 20, 60),
    "chin": (127, 255, 0),
    "left_tibia": (72, 61, 139),
    "right_tibia": (60, 179, 113),
    "lower_spine": (106, 90, 205),
    "upper_half_neck": (199, 21, 133),
    "lower_half_neck": (210, 105, 30),
    "left_chest": (123, 104, 238),
    "right_chest": (85, 107, 47),
    "upper_neck": (47, 79, 79),
    "left_wing_shoulder": (188, 143, 143),
    "left_wing_elbow": (0, 255, 255),
    "right_wing_shoulder": (255, 20, 147),
    "right_wing_elbow": (105, 105, 105),
    "upper_cere": (0, 100, 0),
    "lower_cere": (100, 149, 237),
}

current_index = 0
data = None


def compute_brightness(img, x, y, radius=20):
    """Calculates the average brightness of a region around a point.

    Args:
        img: The input image.
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.
        radius: The radius of the region to consider.

    Returns:
        The average brightness of the region.
    """
    crop = img[
        max(0, y - radius) : min(img.shape[0], y + radius),
        max(0, x - radius) : min(img.shape[1], x + radius),
        :,
    ]
    return np.mean(crop)  # mean brightness


def visualize_coco_annotation(annotation_path, image_dir):
    """Visualizes bounding boxes and keypoints from a COCO annotation file.

    Args:
      annotation_path: Path to the COCO annotation JSON file.
      image_dir: Path to the directory containing the images.
    """
    global data
    with open(annotation_path, "r") as f:
        data = json.load(f)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    fig.canvas.mpl_connect(
        "key_press_event", lambda event: on_key(event, image_dir, ax)
    )
    show_image(image_dir, ax)
    plt.show()


def show_image(image_dir, ax):
    """Shows the current image with bounding boxes and keypoints.

    Args:
      image_dir: Path to the directory containing the images.
      ax: The matplotlib axes to display the image.
    """
    global current_index
    annotation = data["annotations"][current_index]
    image_id = annotation["image_id"]
    image_info = [img for img in data["images"] if img["id"] == image_id][0]
    image_path = os.path.join(image_dir, image_info["file_name"])

    img = cv2.imread(image_path)  # Read image using OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

    # Bounding box visualization
    # TODO currently the code filters based on dataset, from manual inspection, if bbox is visualized from the format [x1, y1, width, height] or [x1, y1, x2, y2] --> Write code that can automatically detect this and apply the right format.
    bbox = annotation["bbox"]
    dataset = image_info.get("source_dataset", "")  # Get dataset information

    if dataset == "cowbird":
        # bbox = [x1, y1, width, height]
        x1, y1, width, height = bbox
        x2, y2 = x1 + width, y1 + height
    else:
        # bbox = [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox

    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Keypoint visualization (if available)
    if "keypoints" in annotation:
        keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)

        for i in range(0, len(keypoints)):
            x1, y1, v = keypoints[i]

            categories = data["categories"][0]
            supercategory_name = categories["name"]
            keypoint_label = categories["keypoints"][i]

            if v > 0:
                cv2.circle(
                    img,
                    center=(int(x1), int(y1)),
                    radius=7,
                    color=COLOR_MAP[keypoint_label],
                    thickness=-1,
                )
                bright = compute_brightness(img, int(x1), int(y1))
                txt_color = (0, 0, 0) if bright > 128 else (255, 255, 255)
                cv2.putText(
                    img,
                    keypoint_label,
                    (int(x1), int(y1) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    txt_color,
                    2,
                )

    ax.imshow(img)
    ax.set_title(f"Category:{supercategory_name}")
    plt.draw()


def on_key(event, image_dir, ax):
    """Handles key press events for navigation.

    Args:
      event: The key press event.
      image_dir: Path to the directory containing the images.
      ax: The matplotlib axes to display the image.
    """
    global current_index
    if event.key == "right":
        current_index = (current_index + 1) % len(data["annotations"])
        print(f"Next image, index: {current_index}")
        ax.clear()
        show_image(image_dir, ax)
    elif event.key == "left":
        current_index = (current_index - 1) % len(data["annotations"])
        print(f"Previous image, index: {current_index}")
        ax.clear()
        show_image(image_dir, ax)


if __name__ == "__main__":
    # replace by your local paths
    # root = "/media/dikra/ADATA HD650/PhD/DATA/DLC24_Data/tiny_all_bird_merged_coco"
    root = (
        "/Users/annastuckert/Documents/DLC_AI_Residency/DLC_AI2024/Bird_datasets_merged"
    )
    # annotation_path = f"{root}/annotations/train.json"
    annotation_path = f"{root}/annotations/test.json"
    image_dir = f"{root}/images"
    visualize_coco_annotation(annotation_path, image_dir)
