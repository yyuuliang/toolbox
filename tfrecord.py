"""

Example of how to create and read tfrecord
"""

# first, prepare the dataset.
# a gt.txt file 
# fname;minx;miny;maxx;maxy;class_id
# and the image files


import hashlib
import io
import logging
import os
import numpy as np
import PIL.Image
import tensorflow as tf
import pandas as pd



# convert data into tfrerocd
def convert_record(data, label_map_dict, image_subdirectory):
        img_path = os.path.join(image_subdirectory, data['filename'])
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

        # here, read the metadata and write into tfrecord
        for obj in data['object']:
                classidx = obj['class']
                xmin.append(float(obj['bndbox']['xmin']) / width)
                ymin.append(float(obj['bndbox']['ymin']) / height)
                xmax.append(float(obj['bndbox']['xmax']) / width)
                ymax.append(float(obj['bndbox']['ymax']) / height)
                class_name = label_map_dict[classidx]
                classes_text.append(class_name.encode('utf8'))
                classes.append(classidx)
        record = tf.train.Example(features=tf.train.Features(feature={


        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),

        'image/object/label/id': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/label/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),

        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        
        'image/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[height,width])),

        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['filename'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpg'.encode('utf8')]))



        }))
        return record


def create_tf_record():
        # labels
        label_map_dict = {1: 'car', 2: 'truck', 3: 'pedestrian', 4: 'trafficLight', 5: 'biker'}
        gt_path = '/home/notus/Whitebase/github/toolbox/tfrecord-images/gt.csv'
        image_dir = '/home/notus/Whitebase/github/toolbox/tfrecord-images/'
        output_filename = '/home/notus/Whitebase/github/toolbox/tfrecord-images/train.record'
        number_total = 5
        num_train = 3
        examples_list = ['%05d.jpg' % x for x in range(number_total)]

        # create the tfrecord file
        writer = tf.python_io.TFRecordWriter(output_filename)

        # read ground truth annotation
        annotations = pd.read_csv(gt_path, delimiter=';', names=('file', 'xMin', 'yMin', 'xMax', 'yMax', 'classId'))


        for idx, example in enumerate(examples_list):
                data = {'filename': example,
                'object': []}
                objects = annotations[annotations['file']==example]
                for _, obj in objects.iterrows():
                        class_id = obj["classId"]
                        data['object'].append({
                                'bndbox': {
                                        'xmin': obj['xMin'],
                                        'ymin': obj['yMin'],
                                        'xmax': obj['xMax'],
                                        'ymax': obj['yMax']
                                },
                                'class': class_id
                                })
                # write one record into the tfrecord
                # this might contains several bounding box info
                tf_record = convert_record(data, label_map_dict, image_dir)
                writer.write(tf_record.SerializeToString())

        writer.close()



"""
Read TF Record
"""

def view_tfrecord():

    # in order to display the data directly
    tf.enable_eager_execution()

    file_path = '/home/notus/Whitebase/github/toolbox/tfrecord-images/train.record'
    filenames = [file_path]
    raw_dataset = tf.data.TFRecordDataset(filenames)

    # Create a description of the features.
    feature_description = {

        # this is a fixed array and size is 1
        'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
        # this is a fixed array and size is 2
        'image/shape': tf.io.FixedLenFeature([2], tf.int64, default_value=[0,0]),
        # we don't know the length of this one, since there might be multiple bounding box in one image, so we use VarLenFeature
        'image/object/label/text': tf.io.VarLenFeature(tf.string),
        'image/object/label/id': tf.io.VarLenFeature(tf.int64)
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)

    idx = 0
    for parsed_record in parsed_dataset.take(3):
        #     print(repr(parsed_record))
        print('image: ', idx)
        idx = idx +1
        print(parsed_record['image/shape'])
        print(parsed_record['image/object/label/text'].values)
        print(parsed_record['image/object/label/id'].values)


if __name__ == '__main__':
        create_tf_record()
        view_tfrecord()
