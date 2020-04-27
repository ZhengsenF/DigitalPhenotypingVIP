from glob import glob
import os
import json
import random
from PIL import Image, ImageDraw
from math import *
from copy import deepcopy


class augmentation:
    # path_to_images: path to image directory
    # path_json: path to label json file
    # path_result: path to directory that stores the result
    def __init__(self, path_to_images, path_json, path_to_results):
        # check if all the input files exist
        if not os.path.isdir(path_to_images):
            raise IOError(f'Directory "{path_to_images}" does not exist')
        if not os.path.isfile(path_json):
            raise IOError(f'annotation file "{path_json}" does not exist')
        if not os.path.isdir(path_to_results):
            os.mkdir(path_to_results)

        self.image_dir = path_to_images
        self.path_json = path_json
        self.layer = []
        self.result = path_to_results

        # read the input files
        self.image_filenames = glob(os.path.join(path_to_images, '*.png'))
        if not self.image_filenames:
            raise IOError(f'Directory "{path_to_images}" does not contain any png image files')
        with open(path_json) as file:
            self.label = json.load(file)

    # add a layer of action to the augmentation
    # example: augmentation.addLayer(rotate_range(0, 180, 0.5))
    def addLayer(self, action):
        self.layer.append(action)

    def __add__(self, other):
        self.addLayer(other)

    # print all the layer of actions
    def printLayer(self):
        print(self.__str__())

    def __str__(self):
        result = ''
        for index, each in enumerate(self.layer):
            result += f'Layer {index} : {each.__str__()}\n'
        return result

    # start augmentation process
    # num_sample: number of sample to be generate
    def augment(self, num_sample):
        if not self.layer:
            raise ValueError('please add layers first before augmentation')
        print('Augment with the following layers:')
        print(self)
        print(f'generating {num_sample} samples')
        image_num = len(self.image_filenames)  # number of images in the directory
        generated_images = []
        generated_image_name = []
        generated_label = {}
        for i in range(num_sample):
            target_image_name = self.image_filenames[i % image_num]
            print(target_image_name)
            target_image = Image.open(target_image_name)
            # update the label
            for each_key in self.label:
                # look for the label for the specific file
                if os.path.split(target_image_name)[1] in each_key:
                    save_name = f'generate{i}.png'  # exported image name
                    generated_label.update({save_name: deepcopy(self.label[each_key])})
                    break
            generated_label[save_name]['filename'] = save_name
            generated_image_name.append(save_name)
            # print(generated_label)
            # go through each layers of transformation
            for each_layer in self.layer:
                target_image, generated_label = each_layer.execute(target_image, generated_label, save_name)
                generated_images.append(target_image)

        # output all the images
        for each_image, each_name in zip(generated_images, generated_image_name):
            each_image.save(os.path.join(self.result, each_name))



# rotate randomly in a angle range
class rotate_range:
    # start_angle: beginning point of the angel range (in degree) (inclusive)
    # end_angle: ending point of the angel range (in degree) (inclusive)
    # probability: the probability that this layer of action will happen (from 0 to 1)
    def __init__(self, start_angle, end_angle, probability):
        if not (0 <= probability <= 1):
            raise ValueError('probability must be within the range from 0 to 1')
        if start_angle < 0:
            raise ValueError('start_angle must be greater than 0 degree')
        if end_angle > 360 or start_angle > 360:
            raise ValueError('end_angle or start-angle must be less than 360 degree')
        if start_angle > end_angle:
            raise ValueError('start_angle must be smaller than or equal to end_angle')
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.probability = probability

    def __str__(self):
        return f'rotate randomly from {self.start_angle} degree to {self.end_angle} degree' \
               f'with probability of {self.probability}'

    def execute(self, image, label, image_name):
        # determine if continue or not
        if not generateBool(self.probability):
            return image, label
        # generate a angle to rotate
        if self.start_angle == self.end_angle:
            angle = self.start_angle
        else:
            angle = generateAngle(self.start_angle, self.end_angle)

        height, width = image.size

        # image_copy = deepcopy(image) #debug
        # draw_copy = ImageDraw.Draw(image_copy) #debug
        # rotate the image
        image = image.rotate(angle)
        draw = ImageDraw.Draw(image)

        # rotate the labeling
        radian = radians(angle)
        for each_key in label:
            # look for the label for the specific file
            if image_name in each_key:
                # rotate the label for each mask
                for region_index, each_region in enumerate(label[each_key]['regions']):
                    x_s = each_region['shape_attributes']['all_points_x']
                    y_s = each_region['shape_attributes']['all_points_y']
                    for index, (each_x, each_y) in enumerate(zip(x_s, y_s)):
                        # draw_copy.point((each_x,each_y)) #debug
                        new_x = (each_x - width / 2) * cos(radian) + (each_y - height / 2) * sin(radian) + width / 2
                        new_y = -(each_x - width / 2) * sin(radian) + (each_y - height / 2) * cos(radian) + height / 2
                        label[each_key]['regions'][region_index]['shape_attributes']['all_points_x'][index] = new_x
                        label[each_key]['regions'][region_index]['shape_attributes']['all_points_y'][index] = new_y

                        draw.point((new_x, new_y))
                break
        # image.show() #debug
        # image_copy.show() #debug
        return image, label


# generate a random angle of transformation
def generateAngle(start, end):
    return random.randrange(start, end)


# generate true or false according to the input probability of being true
def generateBool(probability):
    return random.randrange(100) < (probability * 100)


if __name__ == '__main__':
    a = augmentation(path_to_images='test', path_json='test/test.json', path_to_results='result')
    a.addLayer(rotate_range(50, 70, 0.7))
    a.addLayer(rotate_range(0, 170, 0.4))
    a.addLayer(rotate_range(180, 180, 0.3))
    a.augment(4)
    # a.addLayer(r)
    # a.printLayer()
    # name_test = 'VIS_R_1901385_191206152917120_RGB-Top-0-PNG_101_100_Saturated_191206153625070.png'
    # img = Image.open('test\VIS_R_1901385_191206152917120_RGB-Top-0-PNG_101_100_Saturated_191206153625070.png')
    # with open('test/test.json') as file_test:
    #     label_test = json.load(file_test)
    # r = rotate_range(120, 150, 1)
    # r.execute(img, label_test, name_test)
