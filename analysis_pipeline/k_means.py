import numpy as np
import pandas as pd 

def get_all_keypoints(data):
    ann = len(data["annotations"])
    
    all_keypoints = []
    for i in range(0, ann):
        all_keypoints.append(data["annotations"][i]["keypoints"])
        
    return np.array(all_keypoints)


def parse_keypoints(all_keypoints):
    parsed_keypoints = []
    for k in all_keypoints:
        parsed_keypoints.append(k.reshape((int(k.shape[0]/3), -1)))
    
    return np.array(parsed_keypoints)

def get_keypoints_dict(parsed_keypoints):
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
        index_to_keep = [k for k,v in i.items() if v[2] == 2.]
        visible_keypoints.append({k:v[:2] for k, v in i.items() if k in index_to_keep})
    
    return visible_keypoints

def get_features(visible_keypoints):
    feat = []
    for v in visible_keypoints:
        feat.extend(list(v.values()))
    return feat

def to_df(features, labels):
    df = pd.DataFrame(features, labels)    
    df['label'] = df.index
    
    return df