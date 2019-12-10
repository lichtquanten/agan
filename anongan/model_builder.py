import tensorflow.keras.backend as K
from tensorflow.python.keras.engine.network import Network
from tensorflow.keras.layers import Lambda, Input, Concatenate, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, LeakyReLU, ZeroPadding2D, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta
import tensorflow as tf

def apply_concat_mask(x):
    return tf.concat([x[0] * (1 - x[1]), x[1]], axis=3)

def apply_mask(x):
    return (x[0] * x[2]) + x[1] * (1 - x[2])

def get_models(image_shape):
    optimizer = Adadelta(learning_rate=1.0, rho=0.95)

    mask_shape = (image_shape[0], image_shape[1], 1)
    masked_image_and_mask_shape = (image_shape[0], image_shape[1], image_shape[2] + 1)

    # Inputs
    image = Input(shape=image_shape)
    mask = Input(shape=mask_shape)

    # Completion model
    masked_image_and_mask = Lambda(apply_concat_mask, output_shape=masked_image_and_mask_shape)([image, mask])
    generated_image = build_generator(image_shape, masked_image_and_mask_shape)(masked_image_and_mask)
    completed_image = Lambda(apply_mask, output_shape=image_shape)([generated_image, image, mask])
    completion_model = Model([image, mask], completed_image)
    completion_model.summary()

    # Discriminator model
    validity = build_discriminator(image_shape)(image)
    discriminator_model = Model(image, validity)
    discriminator_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Full model
    discriminator_model.trainable = False
    validity = discriminator_model(completion_model([image, mask]))
    full_model = Model([image, mask], validity)
    full_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    full_model.summary()

    return completion_model, discriminator_model, full_model

def build_discriminator(image_shape):
    g_disc = Sequential()

    g_disc.add(Conv2D(64, kernel_size=5, strides=2, padding="same", input_shape=image_shape))

    g_disc.add(BatchNormalization(momentum=0.8))
    g_disc.add(LeakyReLU(alpha=0.2))

    g_disc.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    g_disc.add(BatchNormalization(momentum=0.8))
    g_disc.add(LeakyReLU(alpha=0.2))

    g_disc.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
    g_disc.add(BatchNormalization(momentum=0.8))
    g_disc.add(LeakyReLU(alpha=0.2))

    for _ in range(3):
        g_disc.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
        g_disc.add(BatchNormalization(momentum=0.8))
        g_disc.add(LeakyReLU(alpha=0.2))

    g_disc.add(Flatten())
    g_disc.add(Dense(1024, activation='relu'))

    g_disc.add(Dense(1, activation='sigmoid'))

    return g_disc

def build_generator(image_shape, masked_image_and_mask_shape):
    model = Sequential()

    # First layer (same dimensions)
    model.add(Conv2D(64, kernel_size=5, strides=1, input_shape=masked_image_and_mask_shape, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # First downsample
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Second Downsample to bottleneck
    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    for _ in range(2):
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

    # dilation layers
    for dilation_rate in [2, 4, 8, 16]:
        model.add(Conv2D(256, kernel_size=3, dilation_rate=dilation_rate, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

    for _ in range(2):
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

    # First upsample
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Second upsample
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(32, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2D(3, kernel_size=3, strides=1, padding="same"))
    model.add(Activation("sigmoid"))

    return model
