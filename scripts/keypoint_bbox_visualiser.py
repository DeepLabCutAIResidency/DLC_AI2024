"""Credits: Gemini"""

import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

# TODO replace keypoint labels by master names from conversion table + consistent colors
# TODO implement skipping images using arrows instead of exit button

BODYPARTS = ["back",
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
"lower_cere"]

COLORS = list(np.random.choice(range(256), size=len(BODYPARTS)))


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
    
    crop = img[max(0, y - radius):min(img.shape[0], y + radius), max(0, x - radius):min(img.shape[1], x + radius),:]
    return np.mean(crop)  # mean brightness


def visualize_coco_annotation(annotation_path, image_dir):
    """Visualizes bounding boxes and keypoints from a COCO annotation file.

    Args:
      annotation_path: Path to the COCO annotation JSON file.
      image_dir: Path to the directory containing the images.
    """
    
    with open(annotation_path, "r") as f:
        data = json.load(f)

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        image_info = [img for img in data["images"] if img["id"] == image_id][0]
        image_path = os.path.join(image_dir, image_info["file_name"])

        img = cv2.imread(image_path)  # Read image using OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

        # Bounding box visualization
        bbox = annotation["bbox"]
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

        plt.imshow(img)
        plt.title(f"Category:{supercategory_name}")
        plt.show()


if __name__ == "__main__":
    # replace by your local paths
    # root = "/media/dikra/PhD/DATA/DLC24_Data/tiny_all_bird_merged_coco"
    root = "/media/dikra/PhD/DATA/DLC24_Data/tiny_nabirds"
    annotation_path = f"{root}/annotations/train.json"
    image_dir = f"{root}/images"
    visualize_coco_annotation(annotation_path, image_dir)
