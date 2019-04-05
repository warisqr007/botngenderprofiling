
import tensorflow as tf

from layers.basics import optimize


class BaseSiameseNet:

    def __init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg, loss_function):
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_len])
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.sentences_lengths = tf.placeholder(dtype=tf.int32, shape=[None])
        self.dropout = 0.2

        self.debug = None

        self.embedding_size = main_cfg['PARAMS'].getint('embedding_size')
        self.learning_rate = main_cfg['TRAINING'].getfloat('learning_rate')

        with tf.variable_scope('embeddings'):
            word_embeddings = tf.get_variable('word_embeddings', [vocabulary_size, self.embedding_size])
            self.embedded_x = tf.gather(word_embeddings, self.x)

        with tf.variable_scope('siamese'):
            self.predictions = self.siamese_layer(max_sequence_len, model_cfg)

        with tf.variable_scope('loss'):
            self.loss = loss_function(self.labels, self.predictions)
            self.opt = optimize(self.loss, self.learning_rate)

        with tf.variable_scope('metrics'):
            self.temp_sim = tf.rint(self.predictions)
            self.correct_predictions = tf.equal(self.temp_sim, tf.to_float(self.labels))
            self.accuracy = tf.reduce_mean(tf.to_float(self.correct_predictions))

        with tf.variable_scope('summary'):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summary_op = tf.summary.merge_all()

    def siamese_layer(self, sequence_len, model_cfg):
        """Implementation of specific siamese layer"""
        raise NotImplementedError()
