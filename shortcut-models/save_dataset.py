import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import csv
from tqdm import tqdm

# Settings
dataset_name = 'celebahq256'
split = 'validation'
output_dir = '/media/embedded/Datasets/Public/CelebAhq256/validation'
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, 'labels.csv')

# TFDS loading and preprocessing
def deserialization_fn(data):
    image = data['image']
    min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
    image = tf.image.resize(image, (256, 256), antialias=True)
    image = tf.cast(image, tf.uint8)  # No normalization, keep as uint8
    return image, data['label']

dataset = tfds.load(dataset_name, split=split)
dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
dataset = tfds.as_numpy(dataset)

# Save images and labels
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'label'])
    for idx, (image, label) in tqdm(enumerate(dataset)):
        img_pil = Image.fromarray(image)
        filename = f"val_{idx:05d}.png"
        img_pil.save(os.path.join(output_dir, filename))
        writer.writerow([filename, int(label)])

print(f"Exported validation set to {output_dir}")