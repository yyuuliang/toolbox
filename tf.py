import numpy as np
import tensorflow as tf


"""
Read TF Record
"""

def view_tfrecord():

    # in order to display the data directly
    tf.enable_eager_execution()

    file_path = 'dataset/lisa/mytfrecord/train.record'
    filenames = [file_path]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    print('raw: ')
    print(raw_dataset)
    idx = 0
    # for raw_record in raw_dataset.take(10):
    #     idx = idx +1
    #     if idx < 10:
    #         print(repr(raw_record))

    # Create a description of the features.
    feature_description = {
        'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/source_id': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/object/class/text': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/object/class/label': tf.FixedLenFeature([], tf.int64, default_value=0),
        'image/object/bbox/xmin': tf.FixedLenFeature([], tf.float32, default_value=0),
        'image/object/bbox/ymin': tf.FixedLenFeature([], tf.float32, default_value=0),
        'image/object/bbox/xmax': tf.FixedLenFeature([], tf.float32, default_value=0),
        'image/object/bbox/ymax': tf.FixedLenFeature([], tf.float32, default_value=0),
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)
    print('parsed: ')
    print(parsed_dataset)

    for parsed_record in parsed_dataset.take(10):
        idx = idx +1
        if idx < 10:
            print(repr(parsed_record))




if __name__ == '__main__':
    view_tfrecord()