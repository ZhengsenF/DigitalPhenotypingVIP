# Author: Zhengsen Fu
# Time: Feb 20 2020
# this script looks for images that are not used in training process
# and copying them into recognition_target directory for mAP assessment

import json
import glob

train_json_fptr = open('D:/maskRCNN/training_set/train/via_region_data.json')
val_json_fptr = open('D:/maskRCNN/training_set/val/via_region_data.json')
train_json = json.load(train_json_fptr)
val_json = json.load(val_json_fptr)
used_image_name = []  # image file names that are used in training process
target_image_name = []  # unused image files

for each_json in [train_json, val_json]:
    for eachImage in each_json:
        used_image_name.append(each_json[eachImage]['filename'])

image_names = glob.glob('D:/maskRCNN/training_images/*')
print(image_names)

train_json_fptr.close()
val_json_fptr.close()
