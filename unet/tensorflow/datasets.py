import os, glob
import tensorflow as tf

def self_driving_dataset(path = '../assets/self_driving/'):
    sub_dir = []

    for file in os.scandir(path):
        if file.is_dir():
            sub_dir.append(file.name)

    images_list = []
    masks_list = []

    for dir in sub_dir:
        images_path = os.path.join(path, dir, dir, "CameraRGB")
        image_list = os.listdir(images_path)
        image_list = [os.path.join(images_path, i) for i in image_list]
        images_list[len(images_list):] = image_list
        
        masks_path = os.path.join(path, dir, dir, "CameraSeg")
        mask_list = os.listdir(masks_path)
        mask_list = [os.path.join(masks_path, i)for i in mask_list]
        masks_list[len(masks_list):] = mask_list


    image_filenames = tf.constant(images_list)
    masks_filenames = tf.constant(masks_list)

    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))

    return dataset

def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')

    return input_image, input_mask

def process_dataset(path):
    dataset = self_driving_dataset(path)
    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(preprocess)

    return processed_image_ds