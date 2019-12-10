from abc import ABC, abstractmethod
import cv2
import numpy as np
import os

import utils

def save_weights(name, model, epoch, postfix):
    dir = os.path.join('weights', postfix)
    if not os.path.exists(dir):
        os.makedirs(dir)
    name += '_epoch_%02d.h5' % epoch
    path = os.path.join(dir, name)
    model.save_weights(path)

class ImageSaver(object):
    def __init__(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.dir = dir

    @staticmethod
    def image_to_str(img):
        img = img * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, img_jpg = cv2.imencode('.jpg', img)
        return img_jpg.tostring()

    def save_image(self, name, img):
        img_str = self.image_to_str(img)
        path = os.path.join(self.dir, name)
        with open(path, 'wb') as f:
            f.write(img_str)
