import json
import os
import glob
import numpy as np
from pprint import pprint as pp
from shutil import copy


def goTroughJson(fileName):
    result = []
    with open(fileName) as file:
        data = json.load(file)
    for key in data:
        each = data[key]
        result.append(each['filename'])
    return result


if __name__ == '__main__':
    train_dir = 'D:/maskRCNN/soybean_training_set/train'  # directory to training set
    val_dir = 'D:/maskRCNN/soybean_training_set/val'  # directory to validation set
    image_dir = 'D:/maskRCNN/soybean_images'  # directory to images
    dirs = [train_dir, val_dir]
    for each_dir in dirs:
        json_path = os.path.join(each_dir, 'via_region_data.json')
        if os.path.exists(json_path):
            FileNotFoundError('json file not exist')
        images = goTroughJson(json_path)
        for each_image in images:
            image_path = os.path.join(image_dir, each_image)
            copy(image_path, each_dir)


