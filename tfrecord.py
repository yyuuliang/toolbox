"""
Example of how to create and read tfrecord
"""

import hashlib
import io
import logging
import os
import numpy as np
import PIL.Image
import tensorflow as tf
import pandas as pd

# convert data into tfrerocd
def convert_example(data, img_path):
        # encode the image
        with tf.gfile.GFile(img_path, 'rb') as fid:
                encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        key = hashlib.sha256(encoded_jpg).hexdigest()

        width, height = image.size
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []

        # there might be multiple bounding boxes
        for obj in data['object']:
                class_id = obj['class_id']
                class_name = obj['class_name']
                classes_text.append(class_name.encode('utf8'))
                classes.append(class_id)
                xmin.append(float(obj['bndbox']['xmin']) / width)
                ymin.append(float(obj['bndbox']['ymin']) / height)
                xmax.append(float(obj['bndbox']['xmax']) / width)
                ymax.append(float(obj['bndbox']['ymax']) / height)

        example = tf.train.Example(features=tf.train.Features(feature={
                # image info
                'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['filename'].encode('utf8')])),
                'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['filename'].encode('utf8')])),
                'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
                'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpg'.encode('utf8')])),
                # this contains encoded image content
                'image/encodedimg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
                # image shape
                'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'image/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[height,width])),
                # bounding box
                'image/object/class/id': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
                'image/object/class/name': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
                'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
                'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
                'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
                'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax))

        }))
        return example


def create_tf_record():
        # labels
        cwd = os.getcwd()
        label_map_dict = {1: 'car', 2: 'truck', 3: 'pedestrian', 4: 'trafficLight', 5: 'biker'}
        gt_path = os.path.join(cwd,'tfrecord-images/gt.csv')
        image_dir = os.path.join(cwd, 'tfrecord-images/')
        output_filename = os.path.join(cwd, 'tfrecord-images/train.record')
        number_total = 3
        examples_list = ['%05d.jpg' % x for x in range(number_total)]

        # create the tfrecord file
        writer = tf.python_io.TFRecordWriter(output_filename)

        # read ground truth annotation
        annotations = pd.read_csv(gt_path, delimiter=';', names=('filename', 'xMin', 'yMin', 'xMax', 'yMax', 'classId'))
        for _, fname in enumerate(examples_list):
                data = {'filename': fname, 'object': []}
                objects = annotations[annotations['filename']==fname]
                for _, obj in objects.iterrows():
                        class_id = obj["classId"]
                        class_name =  label_map_dict[class_id]
                        data['object'].append({
                                'bndbox': {
                                        'xmin': obj['xMin'],
                                        'ymin': obj['yMin'],
                                        'xmax': obj['xMax'],
                                        'ymax': obj['yMax']
                                },
                                'class_id': class_id,
                                'class_name': class_name
                                })
                # write one example into the tfrecord
                img_path = os.path.join(image_dir, data['filename'])
                tf_record = convert_example(data, img_path)
                writer.write(tf_record.SerializeToString())

        writer.close()



"""
Read TF Record
"""

def view_tfrecord():

        # in order to display the data directly
        tf.enable_eager_execution()
        cwd = os.getcwd()

        file_path = os.path.join(cwd, 'tfrecord-images/train.record')
        filenames = [file_path]
        raw_dataset = tf.data.TFRecordDataset(filenames)

        # Create a description of the features.
        feature_description = {

        # this is a fixed array and size is 1
        'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
        # this is a fixed array and size is 2
        'image/shape': tf.io.FixedLenFeature([2], tf.int64, default_value=[0,0]),
        # we don't know the length of this one, since there might be multiple bounding box in one image, so we use VarLenFeature
        'image/object/class/name': tf.io.VarLenFeature(tf.string),
        'image/object/class/id': tf.io.VarLenFeature(tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32)

        }

        # Parse the input tf.Example proto using the dictionary above.
        def _parse_function(example_proto):
                return tf.io.parse_single_example(example_proto, feature_description)

        parsed_dataset = raw_dataset.map(_parse_function)

        idx = 0
        for parsed_record in parsed_dataset.take(3):
                #     print(repr(parsed_record))
                print('image: ', idx)
                idx = idx +1
                print(parsed_record['image/shape'])
                print(parsed_record['image/object/class/name'].values)
                print(parsed_record['image/object/class/id'].values)
                print(parsed_record['image/object/bbox/xmin'].values)
                print(parsed_record['image/object/bbox/ymin'].values)


# Reading a TFRecord file using TF.python_io

def view_tfrecord_io():
        # create a python_io reader
        cwd = os.getcwd()
        file_path = os.path.join(cwd, 'tfrecord-images/train.record')
        record_iterator = tf.python_io.tf_record_iterator(path=file_path)
        idx = 0
        for string_record in record_iterator:
                print('image: ', idx)
                idx = idx +1
                
                example = tf.train.Example()
                example.ParseFromString(string_record)
                print(example.features.feature['image/shape'])
                print(example.features.feature['image/object/class/name'])
                print(example.features.feature['image/object/class/id'])
                print(example.features.feature['image/object/bbox/xmin'])
                print(example.features.feature['image/object/bbox/ymin'])



if __name__ == '__main__':
        # create_tf_record()
        # view_tfrecord()
        # view_tfrecord_io()