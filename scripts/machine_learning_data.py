import os
import json
import requests
import tensorflow as tf
from object_detection.utils import dataset_util

# Path to the JSON annotations file
# json_path = '~/Documents/Project_KMUTNB/scripts/annotations.json'

# Path to the directory containing images
json_path = '~/Documents/Project_KMUTNB/dataset/images/'

def create_tf_example(image_path, annotation):
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Extract annotation information
    label = annotation["name"]
    xmin, ymin, xmax, ymax = extract_polygon_bbox(annotation["polygon"]["path"])

    # Convert to TFExample format
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': dataset_util.bytes_feature(image),
        'image/format': dataset_util.bytes_feature(b'jpeg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature([xmin]),
        'image/object/bbox/xmax': dataset_util.float_list_feature([xmax]),
        'image/object/bbox/ymin': dataset_util.float_list_feature([ymin]),
        'image/object/bbox/ymax': dataset_util.float_list_feature([ymax]),
        'image/object/class/text': dataset_util.bytes_feature(label.encode('utf-8')),
    }))

    return tf_example

def extract_polygon_bbox(polygon_path):
    # Extract bounding box coordinates from the polygon path
    # Implement this function based on the format of your polygon coordinates
    # Calculate xmin, ymin, xmax, ymax based on the polygon coordinates
    pass

# Load JSON annotations
with open(json_path, 'r') as json_file:
    annotations_data = json.load(json_file)

# Create TFRecord file
output_path = 'train.tfrecord'
with tf.io.TFRecordWriter(output_path) as writer:
    for annotation_data in annotations_data:
        image_filename = annotation_data["image"]["filename"]
        image_path = os.path.join(image_dir, image_filename)
        annotations = annotation_data["annotations"]
        for annotation in annotations:
            tf_example = create_tf_example(image_path, annotation)
            writer.write(tf_example.SerializeToString())

print(f"TFRecord file saved at: {output_path}")

