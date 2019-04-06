import logging
import os

import numpy as np
from tflearn.data_utils import VocabularyProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DatasetVectorizer:

    def __init__(self, raw_sentence, model_dir, save_vocab=True):
        os.makedirs(model_dir, exist_ok=True)
        raw_sentence = raw_sentence.ravel()
        raw_sentence = [str(x) for x in list(raw_sentence)]
        self.sentence_length = [len(str(x).split(' ')) for x in list(raw_sentence)]
        max_sentence_length = max(self.sentence_length)
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

    def vectorize_2d(self, raw_sentence):
        #num_instances, num_classes = raw_sentence.shape
        raw_sentence = raw_sentence.ravel()

        for i, v in enumerate(raw_sentence):
            if v is np.nan:
                print(i, v)

        vectorized_sentence = np.array(list(self.vocabulary.transform(raw_sentence)))

        vectorized_sentence = vectorized_sentence.reshape(num_instances, self.max_sentence_len)

        return vectorized_sentence


