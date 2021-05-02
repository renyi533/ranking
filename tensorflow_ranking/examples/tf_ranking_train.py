# Copyright 2021 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""TF Ranking sample code for LETOR datasets in LibSVM format.
WARNING: All data sets are loaded into memory in this sample code. It is
for small data sets whose sizes are < 10G.
A note on the LibSVM format:
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
Sample command lines:
OUTPUT_DIR=/tmp/output && \
TRAIN=tensorflow_ranking/examples/data/train.txt && \
VALI=tensorflow_ranking/examples/data/vali.txt && \
TEST=tensorflow_ranking/examples/data/test.txt && \
rm -rf $OUTPUT_DIR && \
bazel build -c opt \
tensorflow_ranking/examples/tf_ranking_libsvm_py_binary && \
./bazel-bin/tensorflow_ranking/examples/tf_ranking_libsvm_py_binary \
--train_path=$TRAIN \
--vali_path=$VALI \
--test_path=$TEST \
--output_dir=$OUTPUT_DIR \
--num_features=136
You can use TensorBoard to display the training results stored in $OUTPUT_DIR.
Notes:
  * Use --alsologtostderr if the output is not printed into screen.
  * In addition, you can enable multi-objective learning by adding the following
  flags: --secondary_loss=<the secondary loss key>.
"""

from absl import flags

import numpy as np
import six
import tensorflow.compat.v1 as tf
import tensorflow_ranking as tfr
from tensorflow_ranking.python import losses_impl

tf.disable_v2_behavior()

flags.DEFINE_string("train_path", None, "Input file path used for training.")
flags.DEFINE_string("vali_path", None, "Input file path used for validation.")
flags.DEFINE_string("test_path", None, "Input file path used for testing.")
flags.DEFINE_string("output_dir", None, "Output directory for models.")

flags.DEFINE_integer("train_batch_size", 32, "The batch size for training.")
flags.DEFINE_integer("num_train_steps", 10000000000000000, "Number of steps for training.")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs for training.")

flags.DEFINE_float("learning_rate", 0.01, "Learning rate for optimizer.")
flags.DEFINE_float("dropout_rate", 0.5, "The dropout rate before output layer.")
flags.DEFINE_list("hidden_layer_dims", ["256", "128", "64"],
                  "Sizes for hidden layers.")

flags.DEFINE_integer("num_features", 136, "Number of features per document.")
flags.DEFINE_integer("save_checkpoints_steps", 1000, "save_checkpoints_steps")
flags.DEFINE_integer("list_size", 100, "List size used for training.")
flags.DEFINE_integer("group_size", 1, "Group size used in score function.")
flags.DEFINE_string("input_format", 'libsvm', "libsvm or tfrecord.")
flags.DEFINE_string("compression", '', "GZIP/ZLIB.")
flags.DEFINE_bool('enable_lambda', False, 'listwise or pairwise')


flags.DEFINE_string("loss", "pairwise_logistic_loss",
                    "The RankingLossKey for the primary loss function.")
flags.DEFINE_string("discount_func", "log1p",
                    "log1p or linear.")
flags.DEFINE_string("gain_func", "linear",
                    "exp or linear.")
flags.DEFINE_string(
    "secondary_loss", None, "The RankingLossKey for the secondary loss for "
    "multi-objective learning.")
flags.DEFINE_float(
    "secondary_loss_weight", 0.5, "The weight for the secondary loss in "
    "multi-objective learning.")

FLAGS = flags.FLAGS

_PRIMARY_HEAD = "primary_head"
_SECONDARY_HEAD = "secondary_head"

# The document relevance label.
_LABEL_FEATURE = "l"

# Padding labels are set negative so that the corresponding examples can be
# ignored in loss and metrics.
_PADDING_LABEL = -1.

DEFAULT_GAIN_FN = lambda label: label
EXP_GAIN_FN = lambda label: tf.pow(2.0, label) - 1
DEFAULT_RANK_DISCOUNT_FN = lambda rank: tf.math.log(2.) / tf.math.log1p(rank)
RECIPROCAL_RANK_DISCOUNT_FN = lambda rank: 1. / rank


def create_dcg_lambda_weight(topn=None, smooth_fraction=0.):
  """Creates _LambdaWeight for DCG metric."""
  RANK_DISCOUNT_FN = DEFAULT_RANK_DISCOUNT_FN if FLAGS.discount_func == 'log1p' else RECIPROCAL_RANK_DISCOUNT_FN
  GAIN_FN = DEFAULT_GAIN_FN if FLAGS.gain_func == 'linear' else EXP_GAIN_FN
  return losses_impl.DCGLambdaWeight(
      topn,
      gain_fn=GAIN_FN,
      rank_discount_fn=RANK_DISCOUNT_FN,
      normalized=False,
      smooth_fraction=smooth_fraction)

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

def _parse_tfrecord(serialized_example):
    features = {_LABEL_FEATURE :tf.FixedLenFeature([FLAGS.list_size],tf.float32)}
    for fid in range(FLAGS.num_features):
      features[str(fid + 1)] = tf.FixedLenFeature([FLAGS.list_size,1],tf.float32)
    example = tf.parse_single_example(serialized_example,features)
    label = example[_LABEL_FEATURE]
    del example[_LABEL_FEATURE]
    return example, label

# Input Pipeline
def input_fn(path, epochs):
    if FLAGS.input_format == 'libsvm':
      data = tf.data.Dataset.from_generator(
          libsvm_generator(path, FLAGS.num_features, FLAGS.list_size),
          output_types=({str(k): tf.float32 for k in range(1,FLAGS.num_features+1)}, tf.float32),
          output_shapes=(
              {str(k): tf.TensorShape([FLAGS.list_size, 1]) for k in range(1,FLAGS.num_features+1)},
              tf.TensorShape([FLAGS.list_size])
          )
      )
    else:
      d = tf.data.Dataset.list_files(path, shuffle=False, seed=0)
      d = d.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=FLAGS.compression),
          cycle_length=1, block_length=1)
      data = d.map(lambda   x: _parse_tfrecord(x))
      #data = tf.data.TFRecordDataset(path).map(lambda   x: _parse_tfrecord(x))
    if epochs > 1:
      data = data.repeat(epochs)
    data = data.shuffle(max(FLAGS.train_batch_size*5,10000)).batch(FLAGS.train_batch_size).prefetch(1)
  
    return data.make_one_shot_iterator().get_next()

def _use_multi_head():
  """Returns True if using multi-head."""
  return FLAGS.secondary_loss is not None


class IteratorInitializerHook(tf.estimator.SessionRunHook):
  """Hook to initialize data iterator after session is created."""

  def __init__(self):
    super(IteratorInitializerHook, self).__init__()
    self.iterator_initializer_fn = None

  def after_create_session(self, session, coord):
    """Initialize the iterator after the session has been created."""
    del coord
    self.iterator_initializer_fn(session)


def example_feature_columns():
  """Returns the example feature columns."""
  feature_names = ["{}".format(i + 1) for i in range(FLAGS.num_features)]
  return {
      name:
      tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
      for name in feature_names
  }


def load_libsvm_data(path, list_size):
  """Returns features and labels in numpy.array."""

  def _parse_line(line):
    """Parses a single line in LibSVM format."""
    tokens = line.split("#")[0].split()
    assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
    label = float(tokens[0])
    qid = tokens[1]
    kv_pairs = [kv.split(":") for kv in tokens[2:]]
    features = {k: float(v) for (k, v) in kv_pairs}
    return qid, features, label

  tf.compat.v1.logging.info("Loading data from {}".format(path))

  # The 0-based index assigned to a query.
  qid_to_index = {}
  # The number of docs seen so far for a query.
  qid_to_ndoc = {}
  # Each feature is mapped an array with [num_queries, list_size, 1]. Label has
  # a shape of [num_queries, list_size]. We use list for each of them due to the
  # unknown number of queries.
  feature_map = {k: [] for k in example_feature_columns()}
  label_list = []
  total_docs = 0
  discarded_docs = 0
  with open(path, "rt") as f:
    for line in f:
      qid, features, label = _parse_line(line)
      if qid not in qid_to_index:
        # Create index and allocate space for a new query.
        qid_to_index[qid] = len(qid_to_index)
        qid_to_ndoc[qid] = 0
        for k in feature_map:
          feature_map[k].append(np.zeros([list_size, 1], dtype=np.float32))
        label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
      total_docs += 1
      batch_idx = qid_to_index[qid]
      doc_idx = qid_to_ndoc[qid]
      qid_to_ndoc[qid] += 1
      # Keep the first 'list_size' docs only.
      if doc_idx >= list_size:
        discarded_docs += 1
        continue
      for k, v in six.iteritems(features):
        assert k in feature_map, "Key {} not found in features.".format(k)
        feature_map[k][batch_idx][doc_idx, 0] = v
      label_list[batch_idx][doc_idx] = label

  tf.compat.v1.logging.info("Number of queries: {}".format(len(qid_to_index)))
  tf.compat.v1.logging.info(
      "Number of documents in total: {}".format(total_docs))
  tf.compat.v1.logging.info(
      "Number of documents discarded: {}".format(discarded_docs))

  # Convert everything to np.array.
  for k in feature_map:
    feature_map[k] = np.array(feature_map[k])
  return feature_map, np.array(label_list)


def get_train_inputs(features, labels, batch_size):
  """Set up training input in batches."""
  iterator_initializer_hook = IteratorInitializerHook()

  def _train_input_fn():
    """Defines training input fn."""
    features_placeholder = {
        k: tf.compat.v1.placeholder(v.dtype, v.shape)
        for k, v in six.iteritems(features)
    }
    if _use_multi_head():
      placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
      labels_placeholder = {
          _PRIMARY_HEAD: placeholder,
          _SECONDARY_HEAD: placeholder,
      }
    else:
      labels_placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices(
        (features_placeholder, labels_placeholder))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    if _use_multi_head():
      feed_dict = {
          labels_placeholder[head_name]: labels
          for head_name in labels_placeholder
      }
    else:
      feed_dict = {labels_placeholder: labels}
    feed_dict.update(
        {features_placeholder[k]: features[k] for k in features_placeholder})
    iterator_initializer_hook.iterator_initializer_fn = (
        lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
    return iterator.get_next()

  return _train_input_fn, iterator_initializer_hook


def get_eval_inputs(features, labels):
  """Set up eval inputs in a single batch."""
  iterator_initializer_hook = IteratorInitializerHook()

  def _eval_input_fn():
    """Defines eval input fn."""
    features_placeholder = {
        k: tf.compat.v1.placeholder(v.dtype, v.shape)
        for k, v in six.iteritems(features)
    }
    if _use_multi_head():
      placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
      labels_placeholder = {
          _PRIMARY_HEAD: placeholder,
          _SECONDARY_HEAD: placeholder,
      }
    else:
      labels_placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
    dataset = tf.data.Dataset.from_tensors(
        (features_placeholder, labels_placeholder))
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    if _use_multi_head():
      feed_dict = {
          labels_placeholder[head_name]: labels
          for head_name in labels_placeholder
      }
    else:
      feed_dict = {labels_placeholder: labels}
    feed_dict.update(
        {features_placeholder[k]: features[k] for k in features_placeholder})
    iterator_initializer_hook.iterator_initializer_fn = (
        lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
    return iterator.get_next()

  return _eval_input_fn, iterator_initializer_hook


def make_serving_input_fn():
  """Returns serving input fn to receive tf.Example."""
  feature_spec = tf.feature_column.make_parse_example_spec(
      example_feature_columns().values())
  return tf.estimator.export.build_parsing_serving_input_receiver_fn(
      feature_spec)


def make_transform_fn():
  """Returns a transform_fn that converts features to dense Tensors."""

  def _transform_fn(features, mode):
    """Defines transform_fn."""
    if mode == tf.estimator.ModeKeys.PREDICT:
      # We expect tf.Example as input during serving. In this case, group_size
      # must be set to 1.
      if FLAGS.group_size != 1:
        raise ValueError(
            "group_size should be 1 to be able to export model, but get %s" %
            FLAGS.group_size)
      context_features, example_features = (
          tfr.feature.encode_pointwise_features(
              features=features,
              context_feature_columns=None,
              example_feature_columns=example_feature_columns(),
              mode=mode,
              scope="transform_layer"))
    else:
      context_features, example_features = tfr.feature.encode_listwise_features(
          features=features,
          context_feature_columns=None,
          example_feature_columns=example_feature_columns(),
          mode=mode,
          scope="transform_layer")

    return context_features, example_features

  return _transform_fn


def make_score_fn():
  """Returns a groupwise score fn to build `EstimatorSpec`."""

  def _score_fn(unused_context_features, group_features, mode, unused_params,
                unused_config):
    """Defines the network to score a group of documents."""
    with tf.compat.v1.name_scope("input_layer"):
      group_input = [
          tf.compat.v1.layers.flatten(group_features[name])
          for name in sorted(example_feature_columns())
      ]
      input_layer = tf.concat(group_input, 1)
      tf.compat.v1.summary.scalar("input_sparsity",
                                  tf.nn.zero_fraction(input_layer))
      tf.compat.v1.summary.scalar("input_max",
                                  tf.reduce_max(input_tensor=input_layer))
      tf.compat.v1.summary.scalar("input_min",
                                  tf.reduce_min(input_tensor=input_layer))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    cur_layer = tf.compat.v1.layers.batch_normalization(
        input_layer, training=is_training)
    for i, layer_width in enumerate(int(d) for d in FLAGS.hidden_layer_dims):
      cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
      cur_layer = tf.compat.v1.layers.batch_normalization(
          cur_layer, training=is_training)
      cur_layer = tf.nn.relu(cur_layer)
      tf.compat.v1.summary.scalar("fully_connected_{}_sparsity".format(i),
                                  tf.nn.zero_fraction(cur_layer))
    cur_layer = tf.compat.v1.layers.dropout(
        cur_layer, rate=FLAGS.dropout_rate, training=is_training)
    logits = tf.compat.v1.layers.dense(cur_layer, units=FLAGS.group_size)
    if _use_multi_head():
      # Duplicate the logits for both heads.
      return {_PRIMARY_HEAD: logits, _SECONDARY_HEAD: logits}
    else:
      return logits

  return _score_fn

def get_eval_metric_fns():
  """Returns a dict from name to metric functions."""
  RANK_DISCOUNT_FN = DEFAULT_RANK_DISCOUNT_FN if FLAGS.discount_func == 'log1p' else RECIPROCAL_RANK_DISCOUNT_FN
  GAIN_FN = DEFAULT_GAIN_FN if FLAGS.gain_func == 'linear' else EXP_GAIN_FN
  metric_fns = {}
  metric_fns.update({
      "metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name, gain_fn=GAIN_FN, rank_discount_fn=RANK_DISCOUNT_FN) for name in [
          tfr.metrics.RankingMetricKey.ARP,
          tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY,
      ]   
  })  
  metric_fns.update({
      "metric/dcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
          tfr.metrics.RankingMetricKey.DCG, gain_fn=GAIN_FN, topn=topn, rank_discount_fn=RANK_DISCOUNT_FN)
      for topn in [1, 3, 5, 10] 
  })  
  return metric_fns

def train_and_eval():
  """Train and Evaluate."""
  """
  features, labels = load_libsvm_data(FLAGS.train_path, FLAGS.list_size)
  train_input_fn, train_hook = get_train_inputs(features, labels,
                                                FLAGS.train_batch_size)

  features_vali, labels_vali = load_libsvm_data(FLAGS.vali_path,
                                                FLAGS.list_size)
  vali_input_fn, vali_hook = get_eval_inputs(features_vali, labels_vali)

  features_test, labels_test = load_libsvm_data(FLAGS.test_path,
                                                FLAGS.list_size)
  test_input_fn, test_hook = get_eval_inputs(features_test, labels_test)
  """
  optimizer = tf.compat.v1.train.AdagradOptimizer(
      learning_rate=FLAGS.learning_rate)

  def _train_op_fn(loss):
    """Defines train op used in ranking head."""
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    minimize_op = optimizer.minimize(
        loss=loss, global_step=tf.compat.v1.train.get_global_step())
    train_op = tf.group([minimize_op, update_ops])
    return train_op
  
  lambda_weight = None if not FLAGS.enable_lambda else create_dcg_lambda_weight()
  
  if _use_multi_head():
    primary_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(FLAGS.loss, lambda_weight=lambda_weight, 
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS),
        eval_metric_fns=get_eval_metric_fns(),
        train_op_fn=_train_op_fn,
        name=_PRIMARY_HEAD)
    secondary_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(FLAGS.secondary_loss, lambda_weight=lambda_weight, 
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS),
        eval_metric_fns=get_eval_metric_fns(),
        train_op_fn=_train_op_fn,
        name=_SECONDARY_HEAD)
    ranking_head = tfr.head.create_multi_ranking_head(
        [primary_head, secondary_head], [1.0, FLAGS.secondary_loss_weight])
  else:
    ranking_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(FLAGS.loss, lambda_weight=lambda_weight, 
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS),
        eval_metric_fns=get_eval_metric_fns(),
        train_op_fn=_train_op_fn)

  estimator = tf.estimator.Estimator(
      model_fn=tfr.model.make_groupwise_ranking_fn(
          group_score_fn=make_score_fn(),
          group_size=FLAGS.group_size,
          transform_fn=make_transform_fn(),
          ranking_head=ranking_head),
      config=tf.estimator.RunConfig(
          FLAGS.output_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps))

  train_spec = tf.estimator.TrainSpec(
      #input_fn=train_input_fn,
      input_fn=lambda: input_fn(FLAGS.train_path, FLAGS.num_epochs),
      #hooks=[train_hook],
      max_steps=FLAGS.num_train_steps)
  # Export model to accept tf.Example when group_size = 1.
  if FLAGS.group_size == 1:
    vali_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(FLAGS.vali_path, 1),
        #input_fn=vali_input_fn,
        #hooks=[vali_hook],
        steps=1,
        exporters=tf.estimator.LatestExporter(
            "latest_exporter",
            serving_input_receiver_fn=make_serving_input_fn()),
        start_delay_secs=0,
        throttle_secs=30)
  else:
    vali_spec = tf.estimator.EvalSpec(
        #input_fn=vali_input_fn,
        input_fn=lambda: input_fn(FLAGS.vali_path, 1),
        #hooks=[vali_hook],
        steps=1,
        start_delay_secs=0,
        throttle_secs=30)

  tf.compat.v1.logging.info("begin training")
  # Train and validate
  tf.estimator.train_and_evaluate(estimator, train_spec, vali_spec)

  # Evaluate on the test data.
  tf.compat.v1.logging.info("evaluate on training data")
  estimator.evaluate(input_fn=lambda: input_fn(FLAGS.train_path, 1))
  # Evaluate on the test data.
  tf.compat.v1.logging.info("evaluate on validation data")
  estimator.evaluate(input_fn=lambda: input_fn(FLAGS.vali_path, 1))
  # Evaluate on the test data if different from validation data.
  if FLAGS.vali_path != FLAGS.test_path:
    tf.compat.v1.logging.info("evaluate on test data")
    estimator.evaluate(input_fn=lambda: input_fn(FLAGS.test_path, 1))
    #estimator.evaluate(input_fn=test_input_fn, hooks=[test_hook])


def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  train_and_eval()


if __name__ == "__main__":
  flags.mark_flag_as_required("train_path")
  flags.mark_flag_as_required("vali_path")
  flags.mark_flag_as_required("test_path")
  flags.mark_flag_as_required("output_dir")

  tf.compat.v1.app.run()
