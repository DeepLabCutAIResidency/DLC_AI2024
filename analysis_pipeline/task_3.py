"""
Created on Tue Jul 23 14:05:15 2024
@author: mcanela
"""

import json

import matplotlib.pyplot as plt
import numpy as np

folder = "//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/DLC Residency 2024/bird_files/cowbird"

folder_tag = folder.split("/")[-1]
train_file = f"{folder}/annotations/train.json"
test_file = f"{folder}/annotations/test.json"
files = [train_file, test_file]

# =============================================================================
# Identify unusual boxes
# =============================================================================

bbox_widths = []
bbox_heights = []
bbox_areas = []
bbox_aspect_ratios = []

for file in files:
    with open(file, "r") as f:
        coco_data = json.load(f)
        images = coco_data["images"]
        annotations = coco_data["annotations"]
        categories = coco_data["categories"]

        for ann in annotations:
            if "bbox" in ann:
                bbox = ann["bbox"]
                width, height = bbox[2], bbox[3]
                area = width * height
                aspect_ratio = width / height if height > 0 else 0

                bbox_widths.append(width)
                bbox_heights.append(height)
                bbox_areas.append(area)
                bbox_aspect_ratios.append(aspect_ratio)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(folder_tag)  # Set the title for the entire figure

ax1.scatter(bbox_widths, bbox_heights)
ax1.set_title('')
ax1.set_xlabel('Width')
ax1.set_ylabel('Height')

ax2.hist(bbox_areas, bins=50, log=False)
ax2.set_title("")
ax2.set_xlabel("Area (width * height)")
ax2.set_ylabel("Frequency")

ax3.hist(bbox_aspect_ratios, bins=50, log=False)
ax3.set_title("")
ax3.set_xlabel("Ratio (width / height)")
ax3.set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# =============================================================================
# Oculded and non-occluded keypoints
# =============================================================================

keypoints_total = 0
keypoints_unlabeled = 0
keypoints_labeled_occluded = 0
keypoint_labeled_visible = 0

for file in files:
    with open(file, "r") as f:
        coco_data = json.load(f)
        images = coco_data["images"]
        annotations = coco_data["annotations"]
        categories = coco_data["categories"]

        for ann in annotations:
            if "keypoints" in ann:
                keypoints = ann["keypoints"]
                for i in range(0, len(keypoints), 3):
                    visibility = keypoints[i + 2]
                    keypoints_total += 1
                    if visibility == 0:
                        keypoints_unlabeled += 1
                    elif visibility == 1:
                        keypoints_labeled_occluded += 1
                    elif visibility == 2:
                        keypoint_labeled_visible += 1

percent_visible = keypoint_labeled_visible / keypoints_total * 100
percent_occluded = keypoints_labeled_occluded / keypoints_total * 100
percent_unlabeled = keypoints_unlabeled / keypoints_total * 100

percentages = [percent_visible, percent_occluded, percent_unlabeled]
labels = ["Visible (v=2)", "Occluded (v=1)", "Unlabeled (v=0)"]
colors = ["green", "orange", "red"]

plt.figure(figsize=(8, 2))
plt.title(folder_tag)

plt.barh(["Keypoints Visibility"], [percent_visible], color=colors[0], label=labels[0])
plt.barh(
    ["Keypoints Visibility"],
    [percent_occluded],
    left=[percent_visible],
    color=colors[1],
    label=labels[1],
)
plt.barh(
    ["Keypoints Visibility"],
    [percent_unlabeled],
    left=[percent_visible + percent_occluded],
    color=colors[2],
    label=labels[2],
)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().get_yaxis().set_visible(False)

plt.xlabel("% of Keypoints")
plt.legend(loc="upper right")
plt.xlim(0, 100)
plt.tight_layout()
plt.show()
