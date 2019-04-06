from layers.convolution import cnn_layers
from layers.losses import mse
from layers.similarity import manhattan_similarity
from models.base_model import BaseSiameseNet
from utils.config_helpers import parse_list
from layers.basics import dropout
import tensorflow as tf


class CnnSiameseNet(BaseSiameseNet):

    def __init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg):
        BaseSiameseNet.__init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg, mse)

    def siamese_layer(self, sequence_len, model_cfg):
        num_filters = parse_list(model_cfg['PARAMS']['num_filters'])
        filter_sizes = parse_list(model_cfg['PARAMS']['filter_sizes'])

        out = cnn_layers(self.embedded_x,
                          sequence_len,
                          num_filters=num_filters,
                          filter_sizes=filter_sizes)

        
        with tf.name_scope('classifier'):
            L1 = tf.layers.dropout(
                tf.layers.dense(out, 100, activation=tf.nn.relu, name='L1'),
                rate=self.dropout, training=self.is_training)
            y = tf.layers.dense(L1, 1, activation=tf.nn.softmax, name='y')
        return y
