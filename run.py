import time
from argparse import ArgumentParser

import tensorflow as tf
from tqdm import tqdm

from data.dataset import Dataset, DATASETS
from models.model_type import MODELS
from utils.batch_helper import BatchHelper
from utils.config_helpers import MainConfig
from utils.data_utils import DatasetVectorizer
from utils.log_saver import LogSaver
from utils.model_evaluator import ModelEvaluator
from utils.model_saver import ModelSaver
from utils.other_utils import timer, set_visible_gpu, init_config


def train(main_config, model_config, model_name, dataset_name):
    main_cfg = MainConfig(main_config)
    model = MODELS[model_name]
    dataset = DATASETS[dataset_name]()

    model_name = '{}_{}'.format(model_name,
                                main_config['PARAMS']['embedding_size'])

    train_data = dataset.train_set_text()
    vectorizer = DatasetVectorizer(train_data, main_cfg.model_dir)

    dataset_helper = Dataset(vectorizer, dataset, main_cfg.batch_size)
    max_sentence_len = vectorizer.max_sentence_len
    vocabulary_size = vectorizer.vocabulary_size

    train_mini_sen, train_mini_labels = dataset_helper.pick_train_mini_batch()
    train_mini_labels = train_mini_labels.reshape(-1, 1)

    test_sentence = dataset_helper.test_instances()
    test_labels = dataset_helper.test_labels()
    test_labels = test_labels.reshape(-1, 1)

    num_batches = dataset_helper.num_batches
    model = model(max_sentence_len, vocabulary_size, main_config, model_config)
    model_saver = ModelSaver(main_cfg.model_dir, model_name, main_cfg.checkpoints_to_keep)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=main_cfg.log_device_placement)

    with tf.Session(config=config) as session:
        global_step = 0
        init = tf.global_variables_initializer()
        session.run(init)
        log_saver = LogSaver(main_cfg.logs_path, model_name, dataset_name, session.graph)
        model_evaluator = ModelEvaluator(model, session)

        metrics = {'acc': 0.0}
        time_per_epoch = []
        for epoch in tqdm(range(main_cfg.num_epochs), desc='Epochs'):
            start_time = time.time()

            train_sentence = dataset_helper.train_instances(shuffle=True)
            train_labels = dataset_helper.train_labels()

            train_batch_helper = BatchHelper(train_sentence, train_labels, main_cfg.batch_size)

            # small eval set for measuring dev accuracy
            dev_sentence, dev_labels = dataset_helper.dev_instances()
            dev_labels = dev_labels.reshape(-1, 1)

            tqdm_iter = tqdm(range(num_batches), total=num_batches, desc="Batches", leave=False, postfix=metrics)
            for batch in tqdm_iter:
                global_step += 1
                sentence_batch, labels_batch = train_batch_helper.next(batch)
                feed_dict_train = {model.x: sentence_batch,
                                   model.is_training: True,
                                   model.labels: labels_batch}
                loss, _ = session.run([model.loss, model.opt], feed_dict=feed_dict_train)

                if batch % main_cfg.eval_every == 0:
                    feed_dict_train = {model.x: train_mini_sen,
                                     model.is_training: False,
                                     model.labels: train_mini_labels}

                    train_accuracy, train_summary = session.run([model.accuracy, model.summary_op],
                                                                feed_dict=feed_dict_train)
                    log_saver.log_train(train_summary, global_step)

                    feed_dict_dev = {model.x: dev_sentence,
                                     model.is_training: False,
                                     model.labels: dev_labels}

                    dev_accuracy, dev_summary = session.run([model.accuracy, model.summary_op],
                                                            feed_dict=feed_dict_dev)
                    log_saver.log_dev(dev_summary, global_step)
                    tqdm_iter.set_postfix(
                        dev_acc='{:.2f}'.format(float(dev_accuracy)),
                        train_acc='{:.2f}'.format(float(train_accuracy)),
                        loss='{:.2f}'.format(float(loss)),
                        epoch=epoch)

                if global_step % main_cfg.save_every == 0:
                    model_saver.save(session, global_step=global_step)

            model_evaluator.evaluate_dev(dev_sentence, dev_labels)

            end_time = time.time()
            total_time = timer(start_time, end_time)
            time_per_epoch.append(total_time)

            model_saver.save(session, global_step=global_step)

        model_evaluator.evaluate_test(test_sentence, test_labels)
        model_evaluator.save_evaluation('{}/{}'.format(main_cfg.model_dir, model_name), time_per_epoch[-1], dataset)


def predict(main_config, model_config, model):

    model_name = '{}_{}'.format(model,
                                main_config['PARAMS']['embedding_size'])
    model = MODELS[model_name]
    model_dir = str(main_config['DATA']['model_dir'])

    vectorizer = DatasetVectorizer(model_dir)

    max_doc_len = vectorizer.max_sentence_len
    vocabulary_size = vectorizer.vocabulary_size

    model = model(max_doc_len, vocabulary_size, main_config, model_config)

    with tf.Session() as session:
        saver = tf.train.Saver()
        last_checkpoint = tf.train.latest_checkpoint('{}/{}/model'.format(model_dir, model_name))
        saver.restore(session, last_checkpoint)
        while True:
            x = input('Text:')
            x_sen = vectorizer.vectorize(x)

            feed_dict = {model.x: x_sen}
            prediction = session.run([model.temp_sim], feed_dict=feed_dict)
            print(prediction)


def main():
    parser = ArgumentParser()

    parser.add_argument('mode',
                        choices=['train', 'predict'],
                        help='pipeline mode')

    parser.add_argument('model',
                        choices=['rnn', 'cnn', 'multihead', 'bcann', 'bcsann', 'bcsannwmh', 'twolayerbcnn', 'capsann', 'capsnn'],
                        help='model to be used')

    parser.add_argument('dataset',
                        choices=['Gender', 'Bot'],
                        help='dataset to be used')

    parser.add_argument('--gpu',
                        default='0',
                        help='index of GPU to be used (default: %(default))')

    args = parser.parse_args()

    set_visible_gpu(args.gpu)

    main_config = init_config()
    model_config = init_config(args.model)

    mode = args.mode

    if 'train' in mode:
        train(main_config, model_config, args.model, args.dataset)
    else:
        predict(main_config, model_config, args.model)


if __name__ == '__main__':
    main()
