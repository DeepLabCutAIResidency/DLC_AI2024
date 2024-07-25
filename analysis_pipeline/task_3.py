"""
Created on Tue Jul 23 14:05:15 2024
@author: mcanela
"""

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Identify unusual boxes
# =============================================================================

def calculate_width_height(data): 
    bbox_widths = []
    bbox_heights = []   
    annotations = data["annotations"]
    for ann in annotations:
        if "bbox" in ann:
            bbox = ann["bbox"]
            width, height = bbox[2], bbox[3]
            bbox_widths.append(width)
            bbox_heights.append(height)
    return bbox_widths, bbox_heights


def calculate_area(bbox_widths, bbox_heights):
    bbox_areas = []
    for width, height in zip(bbox_widths, bbox_heights):
        area = width * height
        bbox_areas.append(area)
    return bbox_areas


def calculate_ratio(bbox_widths, bbox_heights):
    bbox_aspect_ratios = []
    for width, height in zip(bbox_widths, bbox_heights):
        aspect_ratio = width / height if height > 0 else 0
        bbox_aspect_ratios.append(aspect_ratio)
    return bbox_aspect_ratios


def plot_width_height(bbox_widths, bbox_heights, ax=None): 
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(bbox_widths, bbox_heights)
    ax.set_title('')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    return ax


def plot_area(bbox_areas, ax=None): 
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(bbox_areas, bins=50, log=False)
    ax.set_title("")
    ax.set_xlabel("Area (width * height)")
    ax.set_ylabel("Frequency")
    return ax       
                
                
def plot_ratio(bbox_aspect_ratios, ax=None): 
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(bbox_aspect_ratios, bins=50, log=False)
    ax.set_title("")
    ax.set_xlabel("Ratio (width / height)")
    ax.set_ylabel("Frequency")
    return ax

# =============================================================================
# Oculded and non-occluded keypoints
# =============================================================================

def compute_keypoints(data):
    keypoints_total = 0
    keypoints_unlabeled = 0
    keypoints_labeled_occluded = 0
    keypoint_labeled_visible = 0
    annotations = data["annotations"]
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
    return percent_visible, percent_occluded, percent_unlabeled


def plot_keypoint_percent(percent_visible, percent_occluded, percent_unlabeled, ax=None): 
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 2))
    percentages = [percent_visible, percent_occluded, percent_unlabeled]
    labels = ["Visible (v=2)", "Occluded (v=1)", "Unlabeled (v=0)"]
    colors = ["green", "orange", "red"]
    ax.barh(["Keypoints Visibility"], [percent_visible], color=colors[0], label=labels[0])
    ax.barh(
        ["Keypoints Visibility"],
        [percent_occluded],
        left=[percent_visible],
        color=colors[1],
        label=labels[1],
    )
    ax.barh(
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
    ax.legend(loc="upper right")
    plt.xlim(0, 100)
    plt.tight_layout()
    return ax

