# -*- coding: utf-8 -*-
# @Date    : 2020/12/22
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : multitask_baseline.py
import os
import json
from tqdm import tqdm

from toolkit4nlp.models import *
from toolkit4nlp.layers import *
from toolkit4nlp.utils import *
from toolkit4nlp.optimizers import *
from toolkit4nlp.tokenizers import *
from toolkit4nlp.backend import *

import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report, f1_score


def load_data(filename):
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            label = None
            items = l.strip().split('\t')
            if len(items) == 3:
                idx, text, label = items
            else:
                idx, text = items
            D.append((idx, text, label))
    return D


def load_ocnli_data(filename):
    D = []
    with  open(filename) as f:
        for l in f:
            label = None
            items = l.strip().split('\t')
            if len(items) == 4:
                idx, s1, s2, label = items
            else:
                idx, s1, s2 = items
            D.append((idx, s1, s2, label))
    return D


tnews_train = load_data('/home/mingming.xu/datasets/NLP/ptms_data/TNEWS_train1128.csv')
tnews_test = load_data('/home/mingming.xu/datasets/NLP/ptms_data/TNEWS_a.csv')

ocemotion_train = load_data('/home/mingming.xu/datasets/NLP/ptms_data/OCEMOTION_train1128.csv')
ocemotion_test = load_data('/home/mingming.xu/datasets/NLP/ptms_data/OCEMOTION_a.csv')

ocnli_train = load_ocnli_data('/home/mingming.xu/datasets/NLP/ptms_data/OCNLI_train1128.csv')
ocnli_test = load_ocnli_data('/home/mingming.xu/datasets/NLP/ptms_data/OCNLI_a.csv')


def precess_label(train_data):
    labels = set([d[-1] for d in train_data])
    label2id = {k: v for v, k in enumerate(labels)}
    id2label = {v: k for k, v in label2id.items()}
    return labels, label2id, id2label


tnews_labels, tnews_label2id, tnews_id2label = precess_label(tnews_train)
ocnli_labels, ocnli_label2id, ocnli_id2label = precess_label(ocnli_train)
ocemotion_labels, ocemotion_label2id, ocemotion_id2label = precess_label(ocemotion_train)

tnews_train = [d[:-1] + (tnews_label2id[d[-1]],) for d in tnews_train]
ocnli_train = [d[:-1] + (ocnli_label2id[d[-1]],) for d in ocnli_train]
ocemotion_train = [d[:-1] + (ocemotion_label2id[d[-1]],) for d in ocemotion_train]

# bert config

config_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/vocab.txt'

# tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)

batch_size = 16
maxlen = 256
epochs = 5


class batch_data_generator(DataGenerator):
    def __init__(self, label_mask, **kwargs):
        super(batch_data_generator, self).__init__(**kwargs)
        self.label_mask = label_mask

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels, batch_label_mask = [], [], [], []
        for is_end, item in self.get_sample(shuffle):
            if len(item) == 4:
                _, q, r, l = item
                token_ids, segment_ids = tokenizer.encode(q, r, maxlen=maxlen)
            else:
                _, q, l = item
                token_ids, segment_ids = tokenizer.encode(q, maxlen=maxlen)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([l])
            batch_label_mask.append(self.label_mask)

            if is_end or self.batch_size == len(batch_token_ids):
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_label_mask = pad_sequences(batch_label_mask)
                batch_labels = pad_sequences(batch_labels)

                yield [batch_token_ids, batch_segment_ids, batch_labels, batch_label_mask], None
                batch_token_ids, batch_segment_ids, batch_labels, batch_label_mask = [], [], [], []


split = 0.8
tnews_mask = [1, 0, 0]
ocnli_mask = [0, 1, 0]
ocemotion_mask = [0, 0, 1]


def split_train_valid(data, split):
    n = int(len(data) * split)
    train_data = data[:n]
    valid_data = data[n:]
    return train_data, valid_data


tnews_train_data, tnews_valid_data = split_train_valid(tnews_train, split)
ocnli_train_data, ocnli_valid_data = split_train_valid(ocnli_train, split)
ocemotion_train_data, ocemotion_valid_data = split_train_valid(ocemotion_train, split)

tnews_train_generator = batch_data_generator(data=tnews_train, batch_size=batch_size, label_mask=tnews_mask)
tnews_valid_generator = batch_data_generator(data=tnews_valid_data, batch_size=batch_size, label_mask=tnews_mask)
tnews_test_generator = batch_data_generator(data=tnews_test, batch_size=batch_size, label_mask=tnews_mask)

ocnli_train_generator = batch_data_generator(data=ocnli_train_data, batch_size=batch_size, label_mask=ocnli_mask)
ocnli_valid_generator = batch_data_generator(data=ocnli_valid_data, batch_size=batch_size, label_mask=ocnli_mask)
ocnli_test_generator = batch_data_generator(data=ocnli_test, batch_size=batch_size, label_mask=ocnli_mask)

ocemotion_train_generator = batch_data_generator(data=ocemotion_train_data, batch_size=batch_size,
                                                 label_mask=ocemotion_mask)
ocemotion_valid_generator = batch_data_generator(data=ocemotion_valid_data, batch_size=batch_size,
                                                 label_mask=ocemotion_mask)
ocemotion_test_generator = batch_data_generator(data=ocemotion_test, batch_size=batch_size, label_mask=ocemotion_mask)

train_batch_data = list(tnews_train_generator.__iter__(shuffle=True)) + list(
    ocnli_train_generator.__iter__(shuffle=True))
train_batch_data += list(ocemotion_train_generator.__iter__(shuffle=True))

class data_generator(DataGenerator):
    def __iter__(self, shuffle=False):
        for is_end, item in self.get_sample(shuffle):
            yield item

train_generator = data_generator(data=train_batch_data, batch_size=1)


class SwitchLoss(Loss):
    """计算三种cls 的loss，然后通过 loss mask 过滤掉非当前任务的loss
    这里也可以利用loss mask对不同task 的loss 加权
    """

    def compute_loss(self, inputs, mask=None):
        tnew_pred, ocnli_pred, ocemotion_pred, y_true, type_input = inputs

        train_loss = tf.case(
            [(tf.equal(tf.argmax(type_input[0]), 0), lambda: K.sparse_categorical_crossentropy(y_true, tnews_cls)),
             (tf.equal(tf.argmax(type_input[0]), 1), lambda: K.sparse_categorical_crossentropy(y_true, ocnli_cls)),
             (tf.equal(tf.argmax(type_input[0]), 2), lambda: K.sparse_categorical_crossentropy(y_true, ocemotion_cls))
             ], exclusive=True)
        return K.mean(train_loss)


bert = build_transformer_model(checkpoint_path=checkpoint_path, config_path=config_path, model='nezha')
output = Lambda(lambda x: x[:, 0])(bert.output)
output = Dropout(0.1)(output)

tnews_cls = Dense(units=len(tnews_labels), activation='softmax')(output)
ocnli_cls = Dense(units=len(ocnli_labels), activation='softmax')(output)
ocemotion_cls = Dense(units=len(ocemotion_labels), activation='softmax')(output)

y_input = Input(shape=(None,))
type_input = Input(shape=(None,))

train_output = SwitchLoss(0)([tnews_cls, ocnli_cls, ocemotion_cls, y_input, type_input])

train_model = Model(bert.inputs + [y_input, type_input], train_output)

tnews_model = Model(bert.inputs, tnews_cls)
ocnli_model = Model(bert.inputs, ocnli_cls)
ocemotion_model = Model(bert.inputs, ocemotion_cls)

grad_accum_steps = 3
Opt = extend_with_weight_decay(Adam)
Opt = extend_with_gradient_accumulation(Opt)
exclude_from_weight_decay = ['Norm', 'bias']
Opt = extend_with_piecewise_linear_lr(Opt)
para = {
    'learning_rate': 2e-5,
    'weight_decay_rate': 0.01,
    'exclude_from_weight_decay': exclude_from_weight_decay,
    'grad_accum_steps': grad_accum_steps,
    'lr_schedule': {int(len(train_generator) * 0.1 * epochs / grad_accum_steps): 1,
                    int(len(train_generator) * epochs / grad_accum_steps): 0},
}

opt = Opt(**para)

train_model.compile(opt)


def get_f1(l_t, l_p):
    marco_f1_score = f1_score(l_t, l_p, average='macro')
    return marco_f1_score


def print_result(l_t, l_p):
    marco_f1_score = f1_score(l_t, l_p, average='macro')
    print(marco_f1_score)
    print(f"{'confusion_matrix':*^80}")
    print(confusion_matrix(l_t, l_p, ))
    print(f"{'classification_report':*^80}")
    print(classification_report(l_t, l_p, ))


def get_predict(model, data):
    preds, trues = [], []
    for (t, s, y, _), _ in tqdm(data):
        pred = model.predict([t, s]).argmax(-1)
        preds.extend(pred.tolist())
        trues.extend(y.tolist())
    return trues, preds


def evaluate():
    tnews_trues, tnews_preds = get_predict(tnews_model, tnews_valid_generator)
    ocnli_trues, ocnli_preds = get_predict(ocnli_model, ocnli_valid_generator)
    ocemotion_trues, ocemotion_preds = get_predict(ocemotion_model, ocemotion_valid_generator)

    tnews_f1 = get_f1(tnews_trues, tnews_preds)
    ocnli_f1 = get_f1(ocnli_trues, ocnli_preds)
    ocemotion_f1 = get_f1(ocemotion_trues, ocemotion_preds)

    print_result(tnews_trues, tnews_preds)
    print_result(ocnli_trues, ocnli_preds)
    print_result(ocemotion_trues, ocemotion_preds)

    score = (tnews_f1 + ocnli_f1 + ocemotion_f1) / 3
    return score


class Evaluator(keras.callbacks.Callback):
    def __init__(self, save_path):
        self.save_path = save_path
        self.best_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        avg_f1 = evaluate()
        if self.best_f1 < avg_f1:
            self.best_f1 = avg_f1
            self.model.save_weights(self.save_path)

        print('epoch: {} f1 is:{},  best f1 is:{}'.format(epoch + 1, avg_f1, self.best_f1))





def predict_to_file(result_path):
    _, tnews_preds = get_predict(tnews_model, tnews_test_generator)
    _, ocnli_preds = get_predict(ocnli_model, ocnli_test_generator)
    _, ocemotion_preds = get_predict(ocemotion_model, ocemotion_test_generator)

    tnews_result, ocnli_result, ocemotion_result = [], [], []

    for (d, p) in zip(tnews_test, tnews_preds):
        tnews_result.append({'id': d[0], 'label': tnews_id2label[p]})

    for (d, p) in zip(ocnli_test, ocnli_preds):
        ocnli_result.append({'id': d[0], 'label': ocnli_id2label[p]})

    for (d, p) in zip(ocemotion_test, ocemotion_preds):
        ocemotion_result.append({'id': d[0], 'label': ocemotion_id2label[p]})

    with open(os.path.join(result_path, 'tnews_predict.json'), 'w') as f:
        for d in tnews_result:
            f.write(json.dumps(d) + '\n')

    with open(os.path.join(result_path, 'ocnli_predict.json'), 'w') as f:
        for d in ocnli_result:
            f.write(json.dumps(d) + '\n')

    with open(os.path.join(result_path, 'ocemotion_predict.json'), 'w') as f:
        for d in ocemotion_result:
            f.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    model_save_path = 'best_model.weights'
    evaluator = Evaluator(model_save_path)

    train_model.fit_generator(train_generator.generator(),
                              steps_per_epoch=len(train_generator),
                              epochs=epochs,
                              callbacks=[evaluator]
                              )

    # load best model
    train_model.load_weights(model_save_path)
    # predict to file
    predict_to_file('./result')
