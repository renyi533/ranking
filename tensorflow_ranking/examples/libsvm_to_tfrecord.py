# coding=utf-8
"""
Convert ltr libsvm data to TFRecord
Input: FeatureFile, 
Output: TFRecord binary file

first read group index file

--------------------------------------------------------------------------
Due to the sparse nature of features utilized in most academic datasets for
learning to rank such as LETOR datasets, data points are represented in the
LibSVM format. In this setting, every line encapsulates features and a (graded)
relevance judgment of a query-document pair. The following illustrates the
general structure:
<relevance int> qid:<query_id int> [<feature_id int>:<feature_value float>]
For example:
1 qid:10 32:0.14 48:0.97  51:0.45
0 qid:10 1:0.15  31:0.75  32:0.24  49:0.6
2 qid:10 1:0.71  2:0.36   31:0.58  51:0.12
0 qid:20 4:0.79  31:0.01  33:0.05  35:0.27
3 qid:20 1:0.42  28:0.79  35:0.30  42:0.76
In the above example, the dataset contains two queries. Query "10" has 3
documents, two of which relevant with grades 1 and 2. Similarly, query "20"
has 1 relevant document. Note that query-document pairs may have different
sets of zero-valued features and as such their feature vectors may only
partly overlap or not at all.
--------------------------------------------------------------------------

"""

import os
import sys
import io
import tensorflow.compat.v1 as tf
import threading
import time
import bisect
import multiprocessing
import numpy as np
import six

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
	
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def split_text_file(src, part_count):
    with open(src, "rb") as f:
        file_size = f.seek(0, io.SEEK_END)
        size_per_part = file_size / part_count
        last_position = 0
        for i in range(0, part_count):
            f.seek(int(size_per_part * (i + 1)))
            f.readline()
            position = f.tell()

            f.seek(last_position)
            part_size = position - last_position
            print("Split %d part, start:%d, length:%d" % (i, last_position, part_size))
            with open(src + "_%d" % i, "wb") as part_output:
                left_size = part_size
                while left_size > 0:
                    read_size = 50000000
                    if read_size > left_size:
                        read_size = left_size

                    part_output.write(f.read(read_size))
                    left_size -= read_size
            last_position += part_size

_LABEL_FEATURE='l'
_PADDING_LABEL=-1.0
def _libsvm_parse_line(libsvm_line):
  """Parses a single LibSVM line to a query ID and a feature dictionary.
  Args:
    libsvm_line: (string) input line in LibSVM format.
  Returns:
    A tuple of query ID and a dict mapping from feature ID (string) to value
    (float). "label" is a special feature ID that represents the relevance
    grade.
  """
  tokens = libsvm_line.split()
  qid = int(tokens[1].split(":")[1])
  features = {_LABEL_FEATURE: float(tokens[0])}
  key_values = [key_value.split(":") for key_value in tokens[2:]]
  features.update({key: float(value) for (key, value) in key_values})

  return qid, features

def _libsvm_generate(num_features, list_size, doc_list):
  """Unpacks a list of document features into `Tensor`s.
  Args:
    num_features: An integer representing the number of features per instance.
    list_size: Size of the document list per query.
    doc_list: A list of dictionaries (one per document) where each dictionary is
      a mapping from feature ID (string) to feature value (float).
  Returns:
    A tuple consisting of a dictionary (feature ID to `Tensor`s) and a label
    `Tensor`.
  """
  # Construct output variables.
  features = {}
  for fid in range(num_features):
    features[str(fid + 1)] = np.zeros([list_size, 1], dtype=np.float32)
  labels = np.ones([list_size], dtype=np.float32) * (_PADDING_LABEL)

  # Shuffle the document list and trim to a prescribed list_size.
  np.random.shuffle(doc_list)

  if len(doc_list) > list_size:
    doc_list = doc_list[:list_size]

  # Fill in the output Tensors with feature and label values.
  for idx, doc in enumerate(doc_list):
    for feature_id, value in six.iteritems(doc):
      if feature_id == _LABEL_FEATURE:
        labels[idx] = value
      else:
        features.get(feature_id)[idx, 0] = value

  return features, labels

def libsvm_generator(path, num_features, list_size, seed=None):
  """Parses a LibSVM-formatted input file and aggregates data points by qid.
  Args:
    path: (string) path to dataset in the LibSVM format.
    num_features: An integer representing the number of features per instance.
    list_size: Size of the document list per query.
    seed: Randomization seed used when shuffling the document list.
  Returns:
    A generator function that can be passed to tf.data.Dataset.from_generator().
  """
  if seed is not None:
    np.random.seed(seed)

  def inner_generator():
    """Produces a generator ready for tf.data.Dataset.from_generator.
    It is assumed that data points in a LibSVM-formatted input file are
    sorted by query ID before being presented to this function. This
    assumption simplifies the parsing and aggregation logic: We consume
    lines sequentially and accumulate query-document features until a
    new query ID is observed, at which point the accumulated data points
    are massaged into a tf.data.Dataset compatible representation.
    Yields:
      A tuple of feature and label `Tensor`s.
    """
    # A buffer where observed query-document features will be stored.
    # It is a list of dictionaries, one per query-document pair, where
    # each dictionary is a mapping from a feature ID to a feature value.
    doc_list = []

    with tf.io.gfile.GFile(path, "r") as f:
      # cur indicates the current query ID.
      cur = -1

      for line in f:
        qid, doc = _libsvm_parse_line(line)
        if cur < 0:
          cur = qid

        # If qid is not new store the data and move onto the next line.
        if qid == cur:
          doc_list.append(doc)
          continue

        yield _libsvm_generate(num_features, list_size, doc_list)

        # Reset current pointer and re-initialize document list.
        cur = qid
        doc_list = [doc]

    yield _libsvm_generate(num_features, list_size, doc_list)

  return inner_generator

def convert_to(src, dst, num_features, list_size, compression_type=""):
    generator = libsvm_generator(src, num_features, list_size)
    cnt = 0
    start_stamp = time.time()
    option = tf.io.TFRecordOptions(compression_type=compression_type)
    with tf.io.TFRecordWriter(dst, options=option) as writer:
      for features, labels in generator():
        #print("Writing", dst)
        feature_dict = {}
        #print(features, labels)
        for i in range(num_features):
            fid = str(i+1)
            feature_dict[fid] = _float_list_feature(features.get(fid))
        #feature = {"features" : feature_dict, "labels" : _float_list_feature(labels)}
        feature_dict[_LABEL_FEATURE] = _float_list_feature(labels)
        #print(feature_dict)
        example = tf.train.Example(features = tf.train.Features(feature = feature_dict))
        writer.write(example.SerializeToString())
        cnt = cnt + 1
        echo_freq = 100000
        if cnt % echo_freq == 0:
            print('written %d records with throughput %f ' % (cnt, cnt /  (time.time() - start_stamp)))
    print('written %d records' % cnt)

def main(argv):
    thread_count = multiprocessing.cpu_count()
    
    print("Split input to %d part according to CPU count" % thread_count)
    start_stamp = time.time()
    split_text_file(argv[0], thread_count)
    print("Split file time use: %f seconds" % (time.time() - start_stamp))
    print("Starting %d processors" % thread_count)

    start_stamp = time.time()

    if len(argv) == 4:
      processors = [multiprocessing.Process(target=convert_to, 
          args=("%s_%d" % (argv[0], index), "%s_%d.tfrecord" % (argv[1], index), int(argv[2]), int(argv[3]))) 
          for index in list(range(0, thread_count))]
    elif len(argv) == 5:
      processors = [multiprocessing.Process(target=convert_to, 
          args=("%s_%d" % (argv[0], index), "%s_%d.tfrecord" % (argv[1], index), int(argv[2]), int(argv[3]),argv[4])) 
          for index in list(range(0, thread_count))]
      #convert_to(argv[0], argv[1], int(argv[2]), int(argv[3]), argv[4])
    else:
      print('invaid argument')
    [processor.start() for processor in processors]
    [processor.join() for processor in processors]
    print("Converting timeuse: %f seconds" % (time.time() - start_stamp))

if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit("Usage: ConvertToTFRecord.py data output feature_cnt list_size compresiontype(GZIP/ZLIB/None)") 
    else:
        main(sys.argv[1:])

