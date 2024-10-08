import json 
import pandas as pd
import numpy as np


def load_data(data_path):
    with open(data_path, 'r') as file:
        return json.load(file)
    
def get_all_keypoints(data):
    ann = len(data["annotations"])

    all_keypoints = []
    for i in range(0, ann):
        all_keypoints.append(data["annotations"][i]["keypoints"])

    return np.array(all_keypoints)


def keypoint2triple(all_keypoints):
    '''flat keypoints to triples'''
    parsed_keypoints = []
    for k in all_keypoints:
        parsed_keypoints.append(k.reshape((int(k.shape[0] / 3), -1)))

    return np.array(parsed_keypoints)

def keypoint2pose(all_keypoints):
    return np.where(all_keypoints<2., 0., all_keypoints)


def index_poses(data, parsed_poses):
    all_indexed_poses = dict()
    for i, p in enumerate(parsed_poses):
        all_indexed_poses[i] = p

    return all_indexed_poses


def index_keypoints(parsed_keypoints):
    all_indexed_keypoints = []
    for i, p in enumerate(parsed_keypoints):
        indexed_keypoints = dict()
        for j, pp in enumerate(p):
            indexed_keypoints[j] = pp
        all_indexed_keypoints.append(indexed_keypoints)

    return all_indexed_keypoints


def get_visible_keypoints(indexed_keypoints):
    visible_keypoints = []
    for i in indexed_keypoints:
        index_to_keep = [k for k, v in i.items() if v[2] == 2.0]
        if len(index_to_keep) > 0:
            visible_keypoints.append({k: v[:2] for k, v in i.items() if k in index_to_keep})

    return visible_keypoints

def get_visible_pose(visible_keypoints):
    result = []
    for d in visible_keypoints:
        if d:
            combined_array = np.concatenate(list(d.values()))
            result.append(combined_array)
    return result

def get_features(visible_keypoints):
    feat = []
    for v in visible_keypoints:
        feat.extend(list(v.values()))
    return feat


def to_df(features, labels):
    df = pd.DataFrame(features, labels)
    df["label"] = df.index

    return df
