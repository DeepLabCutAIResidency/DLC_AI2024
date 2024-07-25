

# Function to count keypoints and annotations
def count_keypoints_annotations(data):
    annotations_by_dataset = {}
    keypoints_by_dataset = {}

    for anno in data['annotations']:
        image_id = anno['image_id']
        image_info = next(item for item in data['images'] if item['id'] == image_id)
        dataset = image_info['source_dataset']
        
        if dataset not in annotations_by_dataset:
            annotations_by_dataset[dataset] = 0
            keypoints_by_dataset[dataset] = 0
        annotations_by_dataset[dataset] += 1
        if 'keypoints' in anno:
            keypoints = anno["keypoints"]
            keypoints_labeled = 0
            keypoints_unlabeled = 0
            for i in range(0, len(keypoints), 3):
                visibility = keypoints[i + 2]
                if visibility > 0:
                    keypoints_labeled += 1
                else:
                    keypoints_unlabeled += 1    
            keypoints_by_dataset[dataset] += keypoints_labeled
    return annotations_by_dataset, keypoints_by_dataset


#Currently for cowbird it takes proportion of one bbox out of the entire image, not the sum of all bounding boxes within a single image - see second codeblock for this
def extract_bbox_proportions(data, dataset_name=None):
    proportions = {}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        image = next(img for img in data['images'] if img['id'] == image_id)
        if image['source_dataset']:
            dataset = image['source_dataset']
        else: dataset = dataset_name
        image_size = image['width'] * image['height']
        
        # Set the bbox_size calculation based on dataset
        if dataset == "cowbird":
            # bbox = [x1, y1, width, height]
            bbox_size = bbox[2] * bbox[3]
        else:
            # bbox = [x1, y1, x2, y2]
            bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        proportion = bbox_size / image_size
        if dataset not in proportions:
            proportions[dataset] = []
        proportions[dataset].append(proportion)
    
    return proportions


# Function to calculate proportions of bbox area sum of out parent image size
def sum_bbox_proportions(data, dataset_name=None):
    proportions_by_dataset = {}
    bbox_sums_by_image = {}

    for annotation in data['annotations']:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        image = next(img for img in data['images'] if img['id'] == image_id)
        if image['source_dataset']:
            dataset = image['source_dataset']
        else: dataset = dataset_name
        image_size = image['width'] * image['height']

        # Set the bbox_size calculation based on dataset
        if dataset == "cowbird":
            # bbox = [x1, y1, width, height]
            bbox_size = bbox[2] * bbox[3]
        else:
            # bbox = [x1, y1, x2, y2]
            bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        if image_id not in bbox_sums_by_image:
            bbox_sums_by_image[image_id] = {'sum_bbox_size': 0, 'image_size': image_size, 'dataset': dataset}

        bbox_sums_by_image[image_id]['sum_bbox_size'] += bbox_size

    # Now calculate the proportions for each dataset
    for image_id, bbox_info in bbox_sums_by_image.items():
        proportion = bbox_info['sum_bbox_size'] / bbox_info['image_size']
        dataset = bbox_info['dataset']

        if dataset not in proportions_by_dataset:
            proportions_by_dataset[dataset] = []

        proportions_by_dataset[dataset].append(proportion)

    return proportions_by_dataset