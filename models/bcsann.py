from layers.convolution import cnn_layers
from layers.losses import mse
from layers.similarity import manhattan_similarity
from layers.attention import stacked_multihead_attention
from models.base_model import BaseSiameseNet
from utils.config_helpers import parse_list
from layers.basics import dropout
import tensorflow as tf
import numpy as np

_conv_projection_size = 64
_attention_output_size = 100
_comparison_output_size = 100

class AttentionSCnn(BaseSiameseNet):

    def __init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg):
        BaseSiameseNet.__init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg, mse)
          
        
    def _masked_softmax(self, values, lengths):
        with tf.name_scope('MaskedSoftmax'):
            mask = tf.expand_dims(tf.sequence_mask(lengths, tf.reduce_max(lengths), dtype=tf.float32), -2)
    
            inf_mask = (1 - mask) * -np.inf
            inf_mask = tf.where(tf.is_nan(inf_mask), tf.zeros_like(inf_mask), inf_mask)

            return tf.nn.softmax(tf.multiply(values, mask) + inf_mask)
        
    def _conv_pad(self, values):
        with tf.name_scope('convolutional_padding'):
            pad = tf.zeros([tf.shape(self.x)[0], 1, self.embedding_size])
            return tf.concat([pad, values, pad], axis=1)
        
            
    def siamese_layer(self, sequence_len, model_cfg):
        _conv_filter_size = 3
        #parse_list(model_cfg['PARAMS']['filter_sizes'])
        with tf.name_scope('convolutional_layer'):
            X_conv_1 = tf.layers.conv1d(
                self._conv_pad(self.embedded_x),
                _conv_projection_size,
                _conv_filter_size,
                padding='valid',
                use_bias=False,
                name='conv_1',
            )
            
            
            X_conv_1 = tf.layers.dropout(X_conv_1, rate=self.dropout, training=self.is_training)
            
            X_conv_2 = tf.layers.conv1d(
                self._conv_pad(X_conv_1),
                _conv_projection_size,
                _conv_filter_size,
                padding='valid',
                use_bias=False,
                name='conv_2',
            )
            
            self._X_conv = tf.layers.dropout(X_conv_2, rate=self.dropout, training=self.is_training)
        
            
        with tf.name_scope('self_attention'):
            e_X = tf.layers.dense(self._X_conv, _attention_output_size, activation=tf.nn.relu, name='attention_nn')
            
            e = tf.matmul(e_X, e_X, transpose_b=True, name='e')
            
            self.alfa = tf.matmul(self._masked_softmax(e, sequence_len), self._X_conv, name='alfa') 
            
        with tf.name_scope('comparison_layer'):
            X_comp = tf.layers.dense(
                tf.concat([self._X_conv, self.alfa], 2),
                _comparison_output_size,
                activation=tf.nn.relu,
                name='comparison_nn'
            )
            
            self._X_comp = tf.multiply(
                tf.layers.dropout(X_comp, rate=self.dropout, training=self.is_training),
                tf.expand_dims(tf.sequence_mask(sequence_len, tf.reduce_max(sequence_len), dtype=tf.float32), -1)
            )
            
            self.X_agg = tf.reduce_sum(self._X_comp, 1)
        
        with tf.name_scope('classifier'):
            L1 = tf.layers.dropout(
                tf.layers.dense(self.X_agg, 100, activation=tf.nn.relu, name='L1'),
                rate=self.dropout, training=self.is_training)
            y = tf.layers.dense(L1, 1, activation=tf.nn.softmax, name='y')
        return y
