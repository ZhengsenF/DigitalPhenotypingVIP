from glob import glob
import os
import json
import random
from PIL import Image


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

        # read the input files
        self.image_filenames = glob(os.path.join(path_to_images, '*.png'))
        if not self.image_filenames:
            raise IOError(f'Directory "{path_to_images}" does not contain any png image files')
        with open(path_json) as file:
            self.label = json.load(file)

        self.layer = []

    # add a layer of action to the augmentation
    # example: augmentation.addLayer(rotate_range(augmentation, 0, 180, 0.2))
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
    def augment(self):
        print('augment with the following layers')
        print(self)



# rotate randomly in a angle range
class rotate_range:
    # augmentor: an instance of augmentation
    # start_angle: beginning point of the angel range (in degree) (inclusive)
    # end_angle: ending point of the angel range (in degree) (inclusive)
    # probability: the probability that this layer of action will happen (from 0 to 1)
    def __init__(self, image, label, start_angle, end_angle, probability):
        if not (0 <= probability <= 1):
            raise ValueError('probability must be within the range from 0 to 1')
        if start_angle < 0:
            raise ValueError('start_angle must be greater than 0 degree')
        if end_angle > 360 or start_angle > 360:
            raise ValueError('end_angle or start-angle must be less than 360 degree')
        if start_angle > end_angle:
            raise ValueError('start_angle must be smaller than or equal to end_angle')
        self.image = image
        self.label = label
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.probability = probability

    def __str__(self):
        return f'rotate randomly from {self.start_angle} degree to {self.end_angle} degree' \
               f'with probability of {self.probability}'

    def execute(self):
        if not generateBool(self.probability):
            return self.image, self.label

        if self.start_angle == self.end_angle:
            angle = self.start_angle
        else:
            angle = generateAngle(self.start_angle, self.end_angle)

        self.image = self.image.rotate(angle)


# generate a random angle of transformation
def generateAngle(start, end):
    return random.randrange(start, end)


# generate true or false according to the input probability of being true
def generateBool(probability):
    return random.randrange(100) < (probability * 100)


if __name__ == '__main__':
    a = augmentation(path_to_images='test', path_json='test/test.json', path_to_results='result')
    r = rotate_range([], [], 0, 180, 0.2)
    a.addLayer(r)
    a.addLayer(r)
    a.printLayer()
    img = Image.open('test\VIS_R_1901385_191206152917120_RGB-Top-0-PNG_101_100_Saturated_191206153625070.png')
    r = rotate_range(img, 'test/test.json', 120, 150, 1)
    r.execute()
    print(generateBool(0.2))
    print(generateAngle(2, 180))
