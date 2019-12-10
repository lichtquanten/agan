import numpy as np
import tensorflow as tf

class Trainer(object):
    def __init__(self, batches, generator, discriminator, full, batch_size, log_dir):
        self.batches = batches
        self.generator = generator
        self.discriminator = discriminator
        self.full = full
        self.batch_size = batch_size
        self.batch_idx = 0

        self.tensorboard = tf.keras.callbacks.TensorBoard(
                                              log_dir=log_dir,
                                              write_graph=False,
                                              update_freq='batch')
        self.tensorboard.set_model(full)

        self.real = np.ones(batch_size)
        self.fake = np.zeros(batch_size)

    def train_discriminator(self, num_batches):
        for i in range(num_batches):
            batch = next(self.batches)
            self.train_discriminator_batch(batch)
            self.batch_idx += 1

    def train_generator(self, num_batches):
        for i in range(num_batches):
            batch = next(self.batches)
            self.train_generator_batch(batch)
            self.batch_idx += 1

    def train_both(self, num_batches):
        for i in range(num_batches):
            batch = next(self.batches)
            self.train_discriminator_batch(batch)
            self.train_generator_batch(batch)
            self.batch_idx += 1

    def train_discriminator_batch(self, batch):
        images, masks, masked_images = batch

        generated_images = self.generator.predict([images, masks])

        loss_real, accuracy_real = self.discriminator.train_on_batch([images], self.real)
        loss_fake, accuracy_fake = self.discriminator.train_on_batch([generated_images], self.fake)

        self.tensorboard.on_batch_end(self.batch_idx,   {
                                         'loss_real': loss_real,
                                         'loss_fake': loss_fake,
                                         'accuracy_real': accuracy_real,
                                         'accuracy_fake': accuracy_fake})

        return generated_images

    def train_generator_batch(self, batch):
        images, masks, masked_images = batch

        loss, accuracy = self.full.train_on_batch(
                                    [images, masks], self.real)

        self.tensorboard.on_batch_end(self.batch_idx, {
                                         'loss_adv': loss,
                                         'accuracy_adv': accuracy})

def train(batches, trainer, num_batches, image_dir):
    image_saver = data_saver.ImageSaver(image_dir)

    for i in range(10):
        print('---%d---' % i)
        batch = next(batches)
        generated_images = trainer.train_discriminator(batch)
        trainer.train_generator(batch)
        trainer.batch_idx += 1
