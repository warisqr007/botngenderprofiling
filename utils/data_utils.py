import logging
import os

import numpy as np
from tflearn.data_utils import VocabularyProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DatasetVectorizer:

    def __init__(self, raw_sentence_pairs, model_dir, save_vocab=True):
        os.makedirs(model_dir, exist_ok=True)
        raw_sentence_pairs = raw_sentence_pairs.ravel()
        raw_sentence_pairs = [str(x) for x in list(raw_sentence_pairs)]
        self.sentences_lengths = [len(str(x).split(' ')) for x in list(raw_sentence_pairs)]
        max_sentence_length = max(self.sentences_lengths)
        self.vocabulary = VocabularyProcessor(max_sentence_length)

        if save_vocab:
            self.vocabulary.save('{}/vocab'.format(model_dir))

    @property
    def max_sentence_len(self):
        return self.vocabulary.max_document_length

    @property
    def vocabulary_size(self):
        return len(self.vocabulary.vocabulary_._mapping)

    def restore(self):
        self.vocabulary = VocabularyProcessor.restore('{}/vocab'.format(self.model_dir))

    def vectorize(self, sentence):
        return np.array(list(self.vocabulary.transform([sentence])))

    def vectorize_2d(self, raw_sentence_pairs):
        num_instances, num_classes = raw_sentence_pairs.shape
        raw_sentence_pairs = raw_sentence_pairs.ravel()

        for i, v in enumerate(raw_sentence_pairs):
            if v is np.nan:
                print(i, v)

        vectorized_sentence_pairs = np.array(list(self.vocabulary.transform(raw_sentence_pairs)))

        vectorized_sentence_pairs = vectorized_sentence_pairs.reshape(num_instances, num_classes,
                                                                      self.max_sentence_len)

        vectorized_sentence1 = vectorized_sentence_pairs[:, 0, :]
        vectorized_sentence2 = vectorized_sentence_pairs[:, 1, :]
        return vectorized_sentence1, vectorized_sentence2


