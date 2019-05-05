import tensorflow as tf

from layers.losses import mse
from layers.recurrent import rnn_layer
from layers.similarity import manhattan_similarity
from models.base_model import BaseSiameseNet


class LSTMBasedSiameseNet(BaseSiameseNet):

    def __init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg):
        BaseSiameseNet.__init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg, mse)

    def siamese_layer(self, sequence_len, model_cfg):
        hidden_size = model_cfg['PARAMS'].getint('hidden_size')
        cell_type = model_cfg['PARAMS'].get('cell_type')

        outputs_sen = rnn_layer(self.embedded_x, hidden_size, cell_type)
        
        with tf.name_scope('classifier'):
            L1 = tf.layers.dropout(
                tf.layers.dense(outputs_sen, 100, activation=tf.nn.relu, name='L1'),
                rate=self.dropout, training=self.is_training)
            y = tf.layers.dense(L1, 1, activation=tf.nn.softmax, name='y')
        return y
