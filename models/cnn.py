from layers.convolution import cnn_layers
from layers.losses import mse
from layers.similarity import manhattan_similarity
from models.base_model import BaseSiameseNet
from utils.config_helpers import parse_list
from layers.basics import dropout


class CnnSiameseNet(BaseSiameseNet):

    def __init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg):
        BaseSiameseNet.__init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg, mse)

    def siamese_layer(self, sequence_len, model_cfg):
        num_filters = parse_list(model_cfg['PARAMS']['num_filters'])
        filter_sizes = parse_list(model_cfg['PARAMS']['filter_sizes'])

        out1 = cnn_layers(self.embedded_x1,
                          sequence_len,
                          num_filters=num_filters,
                          filter_sizes=filter_sizes)

        out2 = cnn_layers(self.embedded_x2,
                          sequence_len,
                          num_filters=num_filters,
                          filter_sizes=filter_sizes,
                          reuse=True)

        out1 = dropout(out1, self.is_training)
        out2 = dropout(out2, self.is_training)

        return manhattan_similarity(out1, out2)
