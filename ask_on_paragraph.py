# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ast import literal_eval
from collections import namedtuple
from uuid import uuid1
from time import time

from argparse import ArgumentParser

import collections
import json
import math
import os
import pandas as pd
import random
import re
import modeling
import optimization
import tokenization
import shutil
import six
import tensorflow as tf
import concurrent.futures

tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

from tfidf_closest_docs import TfidfRetriever

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               is_impossible=False):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible)
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               start_position=None,
               end_position=None,
               is_impossible=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible


class DigestContext(object):

  class Retriever(object):

    def __init__(self, context_by_title, tf_idf_top=5):
      self.retriever = TfidfRetriever(min_df=1, max_df=1.0, top_n=tf_idf_top)
      self.retriever.fit(pd.Series(context_by_title))

    def find_closest_docs(self, question):
      return self.retriever.predict(question)


  def __init__(self, context_by_title, top_para_count=5, skip_selection=False):
    self.context_by_title = context_by_title
    self.skip_selection = skip_selection
    if skip_selection:
      top_para_count = len(context_by_title)
    self.retriever = self.Retriever(self.context_by_title, tf_idf_top=top_para_count)

  def is_whitespace(self, c):
    return (c == " " or c == "\t" or c == "\r" or
            c == "\n" or ord(c) == 0x202F)
  
  def process_context(self, context):
    doc_tokens = []
    prev_is_whitespace = True
    for c in context:
      if self.is_whitespace(c):
        prev_is_whitespace = True
      else:
        if prev_is_whitespace:
          doc_tokens.append(c)
        else:
          doc_tokens[-1] += c
        prev_is_whitespace = False
    return " ".join(doc_tokens)

  def process_question(self, question):
    titles, tfidf_scores = self.retriever.find_closest_docs(question)
    for para_title, tfidf_score in zip(titles, tfidf_scores):
      if not tfidf_score and not self.skip_selection:
        break
      yield (para_title, SquadExample(
                qas_id=str(uuid1())[:8],
                question_text=question,
                doc_tokens=self.context_by_title[para_title].split()))


def convert_example_to_features(example, tokenizer, max_seq_length,
                                doc_stride, max_query_length, eval_features,
                                eval_writer, output_fn):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000
  example_index = 0

  query_tokens = tokenizer.tokenize(example.question_text)

  if len(query_tokens) > max_query_length:
    query_tokens = query_tokens[0:max_query_length]

  tok_to_orig_index = []
  all_doc_tokens = []
  for (i, token) in enumerate(example.doc_tokens):
    sub_tokens = tokenizer.tokenize(token)
    for sub_token in sub_tokens:
      tok_to_orig_index.append(i)
      all_doc_tokens.append(sub_token)

  # The -3 accounts for [CLS], [SEP] and [SEP]
  max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

  # We can have documents that are longer than the maximum sequence length.
  # To deal with this we do a sliding window approach, where we take chunks
  # of the up to our max length with a stride of `doc_stride`.
  _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
      "DocSpan", ["start", "length"])
  doc_spans = []
  start_offset = 0
  while start_offset < len(all_doc_tokens):
    length = len(all_doc_tokens) - start_offset
    if length > max_tokens_for_doc:
      length = max_tokens_for_doc
    doc_spans.append(_DocSpan(start=start_offset, length=length))
    if start_offset + length == len(all_doc_tokens):
      break
    start_offset += min(length, doc_stride)

  for (doc_span_index, doc_span) in enumerate(doc_spans):
    tokens = []
    token_to_orig_map = {}
    token_is_max_context = {}
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in query_tokens:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for i in range(doc_span.length):
      split_token_index = doc_span.start + i
      token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

      is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                             split_token_index)
      token_is_max_context[len(tokens)] = is_max_context
      tokens.append(all_doc_tokens[split_token_index])
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    feature = InputFeatures(
        unique_id=unique_id,
        example_index=example_index,
        doc_span_index=doc_span_index,
        tokens=tokens,
        token_to_orig_map=token_to_orig_map,
        token_is_max_context=token_is_max_context,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        is_impossible=example.is_impossible)

    # Run callback
    output_fn(eval_features, eval_writer, feature)

    unique_id += 1


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def create_model(bert_config, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=False,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.compat.v1.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.compat.v1.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  return (start_logits, end_logits)


def model_fn_builder(bert_config, init_checkpoint, use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    (start_logits, end_logits) = create_model(
        bert_config=bert_config,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.compat.v1.trainable_variables()

    initialized_variable_names = {}

    (assignment_map, initialized_variable_names
    ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    
    tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    output_spec = None
    
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
      }
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions)

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.io.FixedLenFeature([], tf.int64),
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
  }


  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=1,
            drop_remainder=drop_remainder))

    return d

  return input_fn


RawResult = collections.namedtuple(
    "RawResult", ["unique_id", "start_logits", "end_logits"])


def predicted_answer_with_logit(
    example, all_features, all_results, n_best_size, max_answer_length,
    version_2_with_negative=False, null_score_diff_threshold=0):

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

  example_index = 0
  features = example_index_to_features[example_index]

  prelim_predictions = []
  # keep track of the minimum score of null start+end of position 0
  score_null = 1000000  # large and positive
  min_null_feature_index = 0  # the paragraph slice with min mull score
  null_start_logit = 0  # the start logit at the slice with min null score
  null_end_logit = 0  # the end logit at the slice with min null score
  for (feature_index, feature) in enumerate(features):
    result = unique_id_to_result[feature.unique_id]
    start_indexes = _get_best_indexes(result.start_logits, n_best_size)
    end_indexes = _get_best_indexes(result.end_logits, n_best_size)
    # if we could have irrelevant answers, get the min score of irrelevant
    if version_2_with_negative:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
          score_null = feature_null_score
          min_null_feature_index = feature_index
          null_start_logit = result.start_logits[0]
          null_end_logit = result.end_logits[0]
    for start_index in start_indexes:
      if start_index >= len(feature.tokens):
        continue
      if start_index not in feature.token_to_orig_map:
        continue
      if not feature.token_is_max_context.get(start_index, False):
        continue
      for end_index in end_indexes:
        # We could hypothetically create invalid predictions, e.g., predict
        # that the start of the span is in the question. We throw out all
        # invalid predictions.
        if end_index >= len(feature.tokens):
          continue
        if end_index not in feature.token_to_orig_map:
          continue
        if end_index < start_index:
          continue
        length = end_index - start_index + 1
        if length > max_answer_length:
          continue
        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=feature_index,
                start_index=start_index,
                end_index=end_index,
                start_logit=result.start_logits[start_index],
                end_logit=result.end_logits[end_index]))
  if version_2_with_negative:
      prelim_predictions.append(
          _PrelimPrediction(
              feature_index=min_null_feature_index,
              start_index=0,
              end_index=0,
              start_logit=null_start_logit,
              end_logit=null_end_logit))
  prelim_predictions = sorted(
      prelim_predictions,
      key=lambda x: (x.start_logit + x.end_logit),
      reverse=True)

  _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "NbestPrediction", ["text", "start_logit", "end_logit"])

  seen_predictions = {}
  nbest = []
  for pred in prelim_predictions:
    if len(nbest) >= n_best_size:
      break
    feature = features[pred.feature_index]
    if pred.start_index > 0:  # this is a non-null prediction
      tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
      orig_doc_start = feature.token_to_orig_map[pred.start_index]
      orig_doc_end = feature.token_to_orig_map[pred.end_index]
      orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
      tok_text = " ".join(tok_tokens)

      # De-tokenize WordPieces that have been split off.
      tok_text = tok_text.replace(" ##", "")
      tok_text = tok_text.replace("##", "")

      # Clean whitespace
      tok_text = tok_text.strip()
      tok_text = " ".join(tok_text.split())
      orig_text = " ".join(orig_tokens)

      final_text = get_final_text(tok_text, orig_text)
      if final_text in seen_predictions:
        continue

      seen_predictions[final_text] = True
    else:
      final_text = ""
      seen_predictions[final_text] = True

    nbest.append(
        _NbestPrediction(
            text=final_text,
            start_logit=pred.start_logit,
            end_logit=pred.end_logit))

  # if we didn't inlude the empty option in the n-best, inlcude it
  if version_2_with_negative:
    if "" not in seen_predictions:
      nbest.append(
          _NbestPrediction(
              text="", start_logit=null_start_logit,
              end_logit=null_end_logit))
  # In very rare edge cases we could have no valid predictions. So we
  # just create a nonce prediction in this case to avoid failure.
  if not nbest:
    nbest.append(
        _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

  assert len(nbest) >= 1

  total_scores = []
  best_non_null_entry = None
  for entry in nbest:
    total_scores.append(entry.start_logit + entry.end_logit)
    if not best_non_null_entry:
      if entry.text:
        best_non_null_entry = entry

  probs = _compute_softmax(total_scores)

  nbest_json = []
  for (i, entry) in enumerate(nbest):
    output = collections.OrderedDict()
    output["text"] = entry.text
    output["probability"] = probs[i]
    output["start_logit"] = entry.start_logit
    output["end_logit"] = entry.end_logit
    nbest_json.append(output)

  assert len(nbest_json) >= 1

  if not version_2_with_negative:
    text = nbest_json[0]["text"]
    logit = nbest_json[0]["start_logit"] + nbest_json[0]["end_logit"]
  else:
    # predict "" iff the null score - the score of best non-null > threshold
    score_diff = score_null - best_non_null_entry.start_logit - (
        best_non_null_entry.end_logit)
    if score_diff > null_score_diff_threshold:
      text = ""
      logit = 0
    else:
      text = best_non_null_entry.text
      logit = best_non_null_entry.start_logit + best_non_null_entry.end_logit

  return (text, logit)


def get_final_text(pred_text, orig_text):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer()

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename):
    self.filename = filename
    self.num_features = 0
    self._writer = tf.io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def validate_args_or_throw(bert_config, max_seq_length, max_query_length):
  """Validate the input args or throw an exception."""

  if max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (max_seq_length, bert_config.max_position_embeddings))

  if max_seq_length <= max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (max_seq_length, max_query_length))

class Chat(object):

  def __init__(
      self, vocab_file, bert_config_file, init_checkpoint,
      max_seq_length, doc_stride, max_answer_length, max_query_length,
      version_2_with_negative, null_score_diff_threshold, n_best_size):

    self.max_seq_length = max_seq_length
    self.doc_stride = doc_stride
    self.max_query_length = max_query_length
    self.max_answer_length = max_answer_length
    self.version_2_with_negative = version_2_with_negative
    self.null_score_diff_threshold = null_score_diff_threshold
    self.n_best_size = n_best_size

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    validate_args_or_throw(
      bert_config, self.max_seq_length, self.max_query_length)
    self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        use_one_hot_embeddings=False)
    self.estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=init_checkpoint.rsplit('/', 1)[0])
    self.contexts_digest = None

  def digest_contexts(self, context_by_title, top_para_count=5, skip_selection=False):
    if not top_para_count:
      top_para_count = 5
    self.contexts_digest = DigestContext(context_by_title=context_by_title,
                                         top_para_count=top_para_count,
                                         skip_selection=skip_selection)

  def append_feature(self, eval_features, eval_writer, feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

  def fetch_answer(self, example):
    output_dir = '/tmp/chat_{}'.format(str(uuid1())[:8])
    os.makedirs(output_dir)
    eval_writer = FeatureWriter(
        filename=os.path.join(output_dir, "eval.tf_record"))
    eval_features = []

    convert_example_to_features(
        example=example,
        tokenizer=self.tokenizer,
        max_seq_length=self.max_seq_length,
        doc_stride=self.doc_stride,
        max_query_length=self.max_query_length,
        eval_features=eval_features,
        eval_writer=eval_writer,
        output_fn=self.append_feature)
    eval_writer.close()
    predict_input_fn = input_fn_builder(
          input_file=eval_writer.filename,
          seq_length=self.max_seq_length,
          drop_remainder=False)

    all_results = []
    start_time = time()

    for result in self.estimator.predict(predict_input_fn, yield_single_examples=True):
      unique_id = int(result["unique_ids"])
      start_logits = [float(x) for x in result["start_logits"].flat]
      end_logits = [float(x) for x in result["end_logits"].flat]
      all_results.append(RawResult(
        unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))

    bot_says, logit = predicted_answer_with_logit(
        example, eval_features, all_results, self.n_best_size,
        self.max_answer_length, self.version_2_with_negative,
        self.null_score_diff_threshold)
    shutil.rmtree(output_dir)
    return {'text': bot_says, 'time_taken': time()-start_time, 'logit': logit}

  def get_answers(self, question):
    # if not question.endswith("?"): question += '?'
    self.answers = []
    if self.contexts_digest:
      for para_title, example in self.contexts_digest.process_question(question):
        answer = self.fetch_answer(example)
        answer['para_title'] = para_title
        if not answer['text'] or answer['text'] == "empty":
          answer['text'] = "Unable to find an answer. Kindly rephrase the question or try another one!"
        answer['logit'] = float("{0:2f}".format(answer['logit']))
        self.answers.append(answer)
    self.answers.sort(key=lambda x: x['logit'], reverse=True)
    return self.answers


def main(model_config_file, bert_architecture, version, model_number, n_best_size=20):

  with open(model_config_file) as f:
    content = json.load(f)

  architectures = list(content.keys())
  if bert_architecture not in architectures:
    raise Exception(
      "Invalid value for bert_architecture.\n"
      "`--bert-architecture` should be one of {}".format(architectures))

  versions = list(content[bert_architecture]['version'].keys())
  if version not in versions:
    raise Exception(
      "Invalid value for version.\n"
      "`--squad-version` should be one of {}".format(versions))

  model_numbers = list(content[bert_architecture]['version'][version].keys())
  if model_number not in model_numbers:
    raise Exception(
      "Invalid value for model_number.\n"
      "`--squad-version` should be one of {}".format(model_numbers))    

  bert_architecture = content[bert_architecture]
  vocab_file = bert_architecture['vocab_file']
  bert_config_file = bert_architecture['bert_config_file']
  
  params = bert_architecture['version'][version][model_number]

  init_checkpoint = \
    os.path.join(
      os.path.dirname(model_config_file),
      version,
      params['directory'],
      params['checkpoint'])

  return Chat(
        vocab_file=bert_architecture['vocab_file'],
        bert_config_file=bert_architecture['bert_config_file'],
        init_checkpoint=init_checkpoint,
        max_seq_length=params["max_seq_length"],
        doc_stride=params['doc_stride'],
        max_answer_length=params['max_answer_length'],
        max_query_length=params['max_query_length'],
        version_2_with_negative=params.get("version_2_with_negative", False),
        null_score_diff_threshold=params.get("null_score_diff_threshold", 0),
        n_best_size=20)



if __name__ == "__main__":

  parser = ArgumentParser()
  parser.add_argument("-mc", "--model-config-file", default='SQUAD_DIR/models/config.json')
  parser.add_argument("-a", "--bert-architecture", default="bert_base_uncased")
  parser.add_argument("-s", "--squad-version", default='v1.1')
  parser.add_argument("-mn", "--model-number", default="0", type=str)
  parser.add_argument("-c", "--context-file", required=True)
  parser.add_argument("--n-best-size", default=20, type=int)
  args = parser.parse_args()

  c = main(args.model_config_file, args.bert_architecture, args.version,
           args.model_number, args.context_file, args.n_best_size)

  question = "What is advising bank?"
  c.get_answers(question)
  

"""
python ask_on_paragraph.py \
  --model-config-file="SQUAD_DIR/models/config.json" \
  --context-file="data/article.txt"
"""