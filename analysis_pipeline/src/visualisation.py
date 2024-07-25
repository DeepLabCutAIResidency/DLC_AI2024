import matplotlib.pyplot as plt

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

