"""
python recognition.py path/to/model path/to/data/directory/ path/to/result/directory/
python recognition.py D:/maskRCNN/DigitalPhenotypingVIP/spike/model/spike.h5 D:/maskRCNN/DigitalPhenotypingVIP/spike/target/ D:/maskRCNN/DigitalPhenotypingVIP/spike/result/
"""

"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from PIL import Image
from PIL import ImageDraw, ImageFont
import glob
import pandas as pd

# Root directory of the project
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith("samples/balloon"):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.model as modellib

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "balloon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + baloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 400

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):
    def load_balloon(self, dataset_dir, subset):

        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("balloon", 1, "balloon")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            print(a['regions'])
            self.add_image(
                "balloon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None, save_path=None):
    assert image_path or video_path
    num_spikes = []
    pixel_count = []
    spike_height = []
    spike_width = []
    center_mask = []
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        print(type(r['masks']))

        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        bbInformationName = os.path.join(save_path, file_name[0:-3] + 'txt')
        # save bb information
        with open(bbInformationName, 'w') as file:
            # <class_name> <left> <top> <right> <bottom> [<difficult>]
            # bb information top left bottom right
            for each_roi, each_score in zip(r['rois'], r['scores']):
                file.write(f'spike {each_score}')
                file.write(f' {each_roi[1]} {each_roi[0]} {each_roi[3]} {each_roi[2]}')
                file.write('\n')

        # draw bb
        spike_cnt = 0  # number of spikes
        for eachBB in r['rois']:
            splash = drawBoundingBox(eachBB, splash)
            spike_cnt += 1
            spike_height.append(eachBB[2] - eachBB[0])
            spike_width.append(eachBB[3] - eachBB[1])
        for _ in range(len(spike_height)):
            num_spikes.append(spike_cnt)

        # draw center of mask by k mean
        for eachBB, maskIndex in zip(r['rois'], range(0, len(r['rois']))):
            topBotList = []
            leftRightList = []
            maskCenter = {}
            for topBot in range(eachBB[0], eachBB[2]):
                for leftRight in range(eachBB[1], eachBB[3]):
                    if r['masks'][topBot][leftRight][maskIndex]:
                        topBotList.append(topBot)
                        leftRightList.append(leftRight)
            maskCenter[maskIndex] = (sum(topBotList) // len(topBotList), sum(leftRightList) // len(leftRightList))
            center_mask.append(f'({maskCenter[maskIndex][0]}, {maskCenter[maskIndex][1]})')
            splash = drawCenterMask(maskCenter[maskIndex], splash)

        # write confidence level
        pilImage = Image.fromarray(splash, 'RGB')
        # fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
        draw = ImageDraw.Draw(pilImage)
        for eachBB, eachText in zip(r['rois'], r['scores']):
            draw.text((eachBB[1], eachBB[0]), '{:3f}'.format(eachText), fill=(255, 255, 255, 255))

        # write number of pixel
        for eachBB, maskIndex in zip(r['rois'], range(0, len(r['rois']))):
            pixelSum = 0
            topBotList = []
            leftRightList = []
            maskCenter = {}
            for topBot in range(eachBB[0], eachBB[2]):
                for leftRight in range(eachBB[1], eachBB[3]):
                    if r['masks'][topBot][leftRight][maskIndex]:
                        pixelSum += 1
                        topBotList.append(topBot)
                        leftRightList.append(leftRight)
            pixel_count.append(pixelSum)
            draw.text((eachBB[1], eachBB[2]), '{}'.format(pixelSum), fill=(255, 255, 255, 255))

        # Save output
        pilImage.save(os.path.join(save_path, file_name))
        # skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)
    return num_spikes, pixel_count, spike_height, spike_width, center_mask


def drawCenterMask(point, imageArr):
    for topDown in range(point[0] - 1, point[0] + 2):
        for leftRight in range(point[1] - 1, point[1] + 2):
            imageArr[topDown][leftRight][0] = 0
            imageArr[topDown][leftRight][1] = 255
            imageArr[topDown][leftRight][2] = 0
    return imageArr


def drawBoundingBox(bb, imageArr):
    # bb = [top left bot right]
    # draw top
    top = bb[0]
    left = bb[1]
    bottom = bb[2]
    right = bb[3]
    bbWidth = 3  # the width of bounding box

    # draw top
    for leftRight in range(left, right):
        for topIndex in range(top - bbWidth, top):
            imageArr[topIndex][leftRight][0] = 0
            imageArr[topIndex][leftRight][1] = 255
            imageArr[topIndex][leftRight][2] = 0

    # draw left
    for leftIndex in range(left, left + bbWidth):
        for topBot in range(top, bottom):
            imageArr[topBot][leftIndex][0] = 0
            imageArr[topBot][leftIndex][1] = 255
            imageArr[topBot][leftIndex][2] = 0

    # draw right
    for rightIndex in range(right - bbWidth, right):
        for topBot in range(top, bottom):
            imageArr[topBot][rightIndex][0] = 0
            imageArr[topBot][rightIndex][1] = 255
            imageArr[topBot][rightIndex][2] = 0

    # draw bottom
    for leftRight in range(left, right):
        for botIndex in range(bottom, bottom + bbWidth):
            imageArr[botIndex][leftRight][0] = 0
            imageArr[botIndex][leftRight][1] = 255
            imageArr[botIndex][leftRight][2] = 0
    return imageArr


############################################################
#  Training
############################################################

if __name__ == '__main__':
    weight = sys.argv[1]
    target = sys.argv[2]
    result = sys.argv[3]

    if not os.path.isfile(weight):
        print('Model file not found!')
        exit(1)
    print(target)
    if not os.path.isdir(target):
        print('Target direct does not exist!')
        exit(1)
    if not os.path.isdir(result):
        print('Creating a result directory')
        os.makedirs(result)

    class InferenceConfig(BalloonConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    config = InferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=DEFAULT_LOGS_DIR)

    # Select weights file to load
    weights_path = weight

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    # detect_and_color_splash(model, image_path=args.image,
    #                         video_path=args.video)
    image_dir = target
    targets = glob.glob(image_dir + '/*.*')
    # csv output
    file_name_main = []
    num_spikes_main = []
    pixel_count_main = []
    spike_height_main = []
    spike_width_main = []
    center_mask_main = []
    save_path = result

    for eachTarget in targets:
        output = detect_and_color_splash(model, image_path=eachTarget, save_path=save_path)
        for _ in range(len(output[0])):
            file_name_main.append(eachTarget)
        # num_spikes_main += output[0]
        for index in range(1, output[0][0] + 1):
            num_spikes_main.append(index)
        pixel_count_main += output[1]
        spike_height_main += output[2]
        spike_width_main += output[3]
        center_mask_main += output[4]

    output_dict = {'file name': file_name_main, 'spike index': num_spikes_main, 'pixel count': pixel_count_main,
                   'spike height': spike_height_main, 'spike width': spike_width_main, 'center of mask': center_mask_main}
    df = pd.DataFrame(output_dict)
    df.to_csv(os.path.join(save_path, 'output.csv'))
