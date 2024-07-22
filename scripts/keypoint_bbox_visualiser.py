"""Credits: Gemini"""

import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


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
    # debug
    print(x, y)
    crop = img[max(0, y - radius):min(img.shape[0], y + radius), max(0, x - radius):min(img.shape[1], x + radius),:]
    return np.mean(crop)  # mean brightness


def visualize_coco_annotation(annotation_path, image_dir):
    """Visualizes bounding boxes and keypoints from a COCO annotation file.

    Args:
      annotation_path: Path to the COCO annotation JSON file.
      image_dir: Path to the directory containing the images.
    """
    
    color_map = {
        "bill": (0, 102, 204),
        "crown": (0, 255, 0),
        "nape": (255, 51, 51),
        "left_eye": (102, 0, 204),
        "right_eye": (255, 255, 102),
        "belly": (255, 128, 0),
        "breast": (0, 255, 255),
        "back": (128, 0, 128),
        "tail": (255, 0, 255),
        "left_wing": (128, 128, 0),
        "right_wing": (0, 128, 128),
    }
    
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
        x, y, w, h = bbox
        cv2.rectangle(img, (int(x), int(y)), (int(w - x), int(h - y)), (0, 255, 0), 2)

        # Keypoint visualization (if available)
        if "keypoints" in annotation:
            keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)

            for i in range(0, len(keypoints)):
                x, y, v = keypoints[i]

                categories = data["categories"][0]
                supercategory_name = categories["name"]
                keypoint_label = categories["keypoints"][i]

                if v > 0:
                    cv2.circle(
                        img,
                        center=(int(x), int(y)),
                        radius=7,
                        color=color_map[keypoint_label],
                        thickness=-1,
                    )
                    bright = compute_brightness(img, int(x), int(y))
                    txt_color = (0, 0, 0) if bright > 128 else (255, 255, 255)
                    cv2.putText(
                        img,
                        keypoint_label,
                        (int(x), int(y) - 15),
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
    # root = "/media/dikra/ADATA HD650/PhD/DATA/DLC24_Data/tiny_all_bird_merged_coco"
    root = "/media/dikra/ADATA HD650/PhD/DATA/DLC24_Data/tiny_nabirds"
    annotation_path = f"{root}/annotations/train.json"
    image_dir = f"{root}/images"
    visualize_coco_annotation(annotation_path, image_dir)
