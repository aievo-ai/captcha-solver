import keras as keras
from numpy import expand_dims


def generate_characters(img, max_batches):
    data = keras.preprocessing.image.img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    generator = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        horizontal_flip=False,
        vertical_flip=False,
        rotation_range=15,
        width_shift_range=.05,
        height_shift_range=.1,)
    batches = 0
    augmented_images = []
    for batch in generator.flow(samples, batch_size=max_batches):
        augmented_images.append(batch[0])
        batches += 1
        if batches >= max_batches:
            break

    return augmented_images
