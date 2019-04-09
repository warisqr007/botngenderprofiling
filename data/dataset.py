from enum import Enum

import numpy as np
import pandas as pd

class DatasetType(Enum):
    Bot = 0,
    Gender = 1


class ColumnType(Enum):
    sentence1 = 0,
    labels = 1


columns = [ColumnType.sentence1.name,
           ColumnType.labels.name]


class DatasetExperiment:

    def __init__(self, dev_ratio=0.01):
        self.data_dir = self._data_path()
        self.dev_ratio = dev_ratio

    def train_set(self):
        raise NotImplementedError

    def train_set_text(self):
        raise NotImplementedError

    def train_labels(self):
        raise NotImplementedError

    def dev_set(self):
        raise NotImplementedError

    def dev_set_text(self):
        raise NotImplementedError

    def dev_labels(self):
        raise NotImplementedError

    def test_set(self):
        raise NotImplementedError

    def test_set_text(self):
        raise NotImplementedError

    def test_labels(self):
        raise NotImplementedError

    def _data_path(self):
        raise NotImplementedError


class BotDataset(DatasetExperiment):

    def __init__(self, *args):
        super().__init__(*args)
        
        self.train = pd.read_csv('{}{}'.format(self.data_dir, 'dataTrainBot.csv'),
                              sep=',',
                              usecols=['Text', 'Human'])
        
        self.test = pd.read_csv('{}{}'.format(self.data_dir, 'dataDevBot.csv'),
                              sep=',',
                              usecols=['Text', 'Human'])
        
        dataset = pd.read_csv('{}{}'.format(self.data_dir, 'dataDevBot.csv'),
                              sep=',',
                              usecols=['Text', 'Human'])
        
        dataset.dropna(inplace=True)
        dataset = dataset.sample(frac=1, random_state=1).reset_index(drop=True)
        num_instances = len(dataset)
        self.num_dev = num_instances * self.dev_ratio
        self.dev = dataset.loc[:self.num_dev]
        

    def train_set(self):
        return self.train

    def train_set_text(self):
        return self.train['Text'].as_matrix()

    def train_labels(self):
        return self.train['Human'].as_matrix()

    def dev_set(self):
        return self.dev

    def dev_set_text(self):
        return self.dev['Text'].as_matrix()

    def dev_labels(self):
        return self.dev['Human'].as_matrix()

    def test_set(self):
        return self.test

    def test_set_text(self):
        return self.test['Text'].as_matrix()

    def test_labels(self):
        return self.test['Human'].as_matrix()

    def _data_path(self):
        return 'corpora/Bot/'


    
class GenderDataset(DatasetExperiment):

    def __init__(self, *args):
        super().__init__(*args)
        
        self.train = pd.read_csv('{}{}'.format(self.data_dir, 'dataTrainGender.csv'),
                              sep=',',
                              usecols=['Text', 'Male'])
        
        self.test = pd.read_csv('{}{}'.format(self.data_dir, 'dataDevGender.csv'),
                              sep=',',
                              usecols=['Text', 'Male'])
        
        dataset = pd.read_csv('{}{}'.format(self.data_dir, 'dataDevGender.csv'),
                              sep=',',
                              usecols=['Text', 'Male'])
        
        dataset.dropna(inplace=True)
        dataset = dataset.sample(frac=1, random_state=1).reset_index(drop=True)
        num_instances = len(dataset)
        self.num_dev = num_instances * self.dev_ratio
        self.dev = dataset.loc[:self.num_dev]
        

    def train_set(self):
        return self.train

    def train_set_text(self):
        return self.train['Text'].as_matrix()

    def train_labels(self):
        return self.train['Male'].as_matrix()

    def dev_set(self):
        return self.dev

    def dev_set_text(self):
        return self.dev['Text'].as_matrix()

    def dev_labels(self):
        return self.dev['Male'].as_matrix()

    def test_set(self):
        return self.test

    def test_set_text(self):
        return self.test['Text'].as_matrix()

    def test_labels(self):
        return self.test['Male'].as_matrix()

    def _data_path(self):
        return 'corpora/Gender/'



DATASETS = {
    DatasetType.Bot.name: BotDataset,
    DatasetType.Gender.name: GenderDataset
}


class Dataset:

    def __init__(self, vectorizer, dataset, batch_size):

        self.train_sen = vectorizer.vectorize_2d(dataset.train_set_text())
        self.dev_sen = vectorizer.vectorize_2d(dataset.dev_set_text())
        self.test_sen = vectorizer.vectorize_2d(dataset.test_set_text())
        self.num_tests = len(dataset.test_set())
        self._train_labels = dataset.train_labels()
        self._dev_labels = dataset.dev_labels()
        self._test_labels = dataset.test_labels()
        self.__shuffle_train_idxs = range(len(self._train_labels))
        self.num_batches = len(self._train_labels) // batch_size

    def train_instances(self, shuffle=False):
        if shuffle:
            self.__shuffle_train_idxs = np.random.permutation(range(len(self.__shuffle_train_idxs)))
            self.train_sen = self.train_sen[self.__shuffle_train_idxs]
            self._train_labels = self._train_labels[self.__shuffle_train_idxs]
        return self.train_sen

    def train_labels(self):
        return self._train_labels

    def test_instances(self):
        return self.test_sen

    def test_labels(self):
        return self._test_labels

    def dev_instances(self):
        return self.dev_sen, self._dev_labels

    def num_dev_instances(self):
        return len(self._dev_labels)

    def pick_train_mini_batch(self):
        train_idxs = np.arange(len(self._train_labels))
        np.random.shuffle(train_idxs)
        train_idxs = train_idxs[:self.num_dev_instances()]
        return self.train_sen[train_idxs], self._train_labels[train_idxs]

    def __str__(self):
        return 'Dataset properties:\n ' \
               'Number of training instances: {}\n ' \
               'Number of dev instances: {}\n ' \
               'Number of test instances: {}\n' \
            .format(len(self._train_labels), len(self._dev_labels), len(self._test_labels))
