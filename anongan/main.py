import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

import data_loader, model_builder
from trainer import Trainer, train
import data_saver

def get_dirs(postfix):
    dirs = []
    for name in ['images', 'logs', 'weights']:
        dir = os.path.join(name, postfix)
        dirs.append(dir)
        if not os.path.exists(dir):
            os.makedirs(dir)
    return dirs

def main(config):
    # Create dataset
    dataset = data_loader.ImageDataset(
                                image_width=config['image_width'],
                                image_height=config['image_height'])
    batches = data_loader.dataset_to_batches(dataset,
                                batch_size=config['batch_size'],
                                prefetch_size=5)


    # Create models
    image_shape = (config['image_width'], config['image_height'], 3)
    generator, discriminator, full = model_builder.get_models(image_shape)

    # Load weights
    if config['generator_weights']:
        generator.load_weights(config['generator_weights'])

    if config['discriminator_weights']:
        discriminator.load_weights(config['discriminator_weights'])

    if config['full_weights']:
        full.load_weights(config['full_weights'])

    # Create output directories
    postfix = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    image_dir, log_dir, weights_dir = get_dirs(postfix)

    # Train
    trainer = Trainer(generator, discriminator, full, config['batch_size'], log_dir)
    train(batches, trainer, config['num_batches'], image_dir)

    # Save weights
    if config['save_weights']:
        data_saver.save_weights('generator', generator, epoch, postfix)
        data_saver.save_weights('discriminator', discriminator, epoch, postfix)
        data_saver.save_weights('full', full, epoch, postfix)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Image configuration
    parser.add_argument('--image-width', type=int, default=256)
    parser.add_argument('--image-height', type=int, default=256)

    # Training configuration
    parser.add_argument('--batch-size', type=int, default=18)
    parser.add_argument('--num-batches', type=int, default=18)

    # Load existing weights
    parser.add_argument('--generator-weights', type=str, default='')
    parser.add_argument('--discriminator-weights', type=str, default='')
    parser.add_argument('--full-weights', type=str, default='')

    # Outputs
    parser.add_argument('--save-weights', type=str2bool, default=False)

    config = vars(parser.parse_args())
    main(config)
