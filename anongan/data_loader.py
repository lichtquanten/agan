from abc import ABC, abstractmethod
import cv2
import numpy as np
import random
import tensorflow as tf

def get_random_color():
    return (
        random.randint(0,256),
        random.randint(0,256),
        random.randint(0,256)
    )

def dataset_to_batches(dataset, batch_size=16, prefetch_size=5):
    dataset = tf.data.Dataset.from_generator(dataset, (tf.float32, tf.int32, tf.float32))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)
    batches = iter(dataset)
    batches = (tuple(map(lambda y: y.numpy(), x)) for x in batches)
    return batches

class ImageCreator(object):
    def __init__(self, image_shape):
        self.image_shape = image_shape

    def create_image(self):
        image = np.zeros(self.image_shape, np.uint8)
        background_color = get_random_color()
        image[:] = background_color
        return image

class MaskCreator(object):
    def __init__(self, image_shape):
        self.mask_shape = image_shape[:2] + (1,)

    def create_mask(self):
        mask = np.zeros(self.mask_shape, np.uint8)
        mask = cv2.rectangle(mask, (10,10), (50, 50), color=1, thickness=-1)
        return mask

class ImageDataset(object):
    def __init__(self, image_width=256, image_height=256):
        self.image_shape = (image_width, image_height, 3)
        self.image_creator = ImageCreator(self.image_shape)
        self.mask_creator = MaskCreator(self.image_shape)

    @staticmethod
    def normalize(image):
        image = image / float(255)
        image = image.astype(np.float32)
        return image

    def __call__(self):
        return self.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        image = self.image_creator.create_image()
        image = self.normalize(image)
        mask = self.mask_creator.create_mask()
        masked_image = cv2.bitwise_and(image, image, mask=(mask+1)%2)
        return image, mask, masked_image
