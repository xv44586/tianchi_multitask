# -*- coding: utf-8 -*-
# @Date    : 2020/12/23
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : pretraining.py
import os
import numpy as np

os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

from toolkit4nlp.models import *
from toolkit4nlp.layers import *
from toolkit4nlp.utils import *
from toolkit4nlp.optimizers import *
from toolkit4nlp.tokenizers import *
from toolkit4nlp.backend import *
import tensorflow as tf
import jieba

jieba.initialize()
# bert config
config_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/vocab.txt'

# tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)

batch_size = 14
maxlen = 325
epochs = 50
learning_rate = 5e-5


def load_data(filename):
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            label = None
            items = l.strip().split('\t')
            if len(items) == 3:
                idx, text, label = items
            #                 label = int(label)
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

new_dict_path = 'new_dict.txt'

new_words = []
with open(new_dict_path) as f:
    for l in f:
        w = l.strip()
        new_words.append(w)
        jieba.add_word(w)

tnews_words = [[jieba.lcut(line)] for line in tnews]
ocnli_words = [[jieba.lcut(line) for line in lines] for lines in ocnli]
ocemotion_words = [[jieba.lcut(line)] for line in ocemotion]


def can_mask(token_ids):
    if token_ids in (tokenizer._token_start_id, tokenizer._token_mask_id, tokenizer._token_end_id):
        return False

    return True


def random_masking(lines):
    """对输入进行随机mask
    """

    if type(lines[0]) != list:
        lines = [lines]

    sources, targets = [tokenizer._token_start_id], [0]
    segments = [0]

    for i, sent in enumerate(lines):
        source, target = [], []
        segment = []
        rands = np.random.random(len(sent))
        for r, word in zip(rands, sent):
            word_token = tokenizer.encode(word)[0][1:-1]

            if r < 0.15 * 0.8:
                source.extend(len(word_token) * [tokenizer._token_mask_id])
                target.extend(word_token)
            elif r < 0.15 * 0.9:
                source.extend(word_token)
                target.extend(word_token)
            elif r < 0.15:
                source.extend([np.random.choice(tokenizer._vocab_size - 5) + 5 for _ in range(len(word_token))])
                target.extend(word_token)
            else:
                source.extend(word_token)
                target.extend([0] * len(word_token))

        # add end token
        source.append(tokenizer._token_end_id)
        #         target.append(tokenizer._token_end_id)
        target.append(0)

        if i == 0:
            segment = [0] * len(source)
        else:
            segment = [1] * len(source)

        sources.extend(source)
        targets.extend(target)
        segments.extend(segment)

    return sources, targets, segments


class data_generator(DataGenerator):
    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked, batch_nsp = [], [], [], [], []

        for is_end, item in self.get_sample(shuffle):
            source_tokens, target_tokens, segment_ids = random_masking(item)

            is_masked = [0 if i == 0 else 1 for i in target_tokens]
            batch_token_ids.append(source_tokens)
            batch_segment_ids.append(segment_ids)
            batch_target_ids.append(target_tokens)
            batch_is_masked.append(is_masked)
            #             batch_nsp.append([label])

            if is_end or len(batch_token_ids) == self.batch_size:
                batch_token_ids = pad_sequences(batch_token_ids, maxlen=maxlen)
                batch_segment_ids = pad_sequences(batch_segment_ids, maxlen=maxlen)
                batch_target_ids = pad_sequences(batch_target_ids, maxlen=maxlen)
                batch_is_masked = pad_sequences(batch_is_masked, maxlen=maxlen)
                #                 batch_nsp = pad_sequences(batch_nsp)

                yield [batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked], None

                batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked = [], [], [], []


#                 batch_nsp = []

train_data_generator = data_generator(tnews_words + ocnli_words + ocemotion_words, batch_size=batch_size)


def build_transformer_model_with_mlm():
    """带mlm的bert模型
    """
    bert = build_transformer_model(
        config_path,
        with_mlm='linear',
        #         with_nsp=True,
        model='nezha',
        return_keras_model=False,
        #         keep_tokens=keep_tokens
    )
    proba = bert.model.output
    # 辅助输入
    token_ids = Input(shape=(None,), dtype='int64', name='token_ids')  # 目标id
    is_masked = Input(shape=(None,), dtype=K.floatx(), name='is_masked')  # mask标记

    #     nsp_label = Input(shape=(None,), dtype='int64', name='nsp')  # nsp

    def mlm_loss(inputs):
        """计算loss的函数，需要封装为一个层
        """
        y_true, y_pred, mask = inputs
        #         _, y_pred = y_pred
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * mask) / (K.sum(mask) + K.epsilon())
        return loss

    def nsp_loss(inputs):
        """计算nsp loss的函数，需要封装为一个层
        """
        y_true, y_pred = inputs
        y_pred, _ = y_pred
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred
        )
        loss = K.mean(loss)
        return loss

    def mlm_acc(inputs):
        """计算准确率的函数，需要封装为一个层
        """
        y_true, y_pred, mask = inputs
        #         _, y_pred = y_pred
        y_true = K.cast(y_true, K.floatx())
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * mask) / (K.sum(mask) + K.epsilon())
        return acc

    def nsp_acc(inputs):
        """计算准确率的函数，需要封装为一个层
        """
        y_true, y_pred = inputs
        y_pred, _ = y_pred
        y_true = K.cast(y_true, K.floatx())
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.mean(acc)
        return acc

    mlm_loss = Lambda(mlm_loss, name='mlm_loss')([token_ids, proba, is_masked])
    mlm_acc = Lambda(mlm_acc, name='mlm_acc')([token_ids, proba, is_masked])
    #     nsp_loss = Lambda(nsp_loss, name='nsp_loss')([nsp_label, proba])
    #     nsp_acc = Lambda(nsp_acc, name='nsp_acc')([nsp_label, proba])

    train_model = Model(
        bert.model.inputs + [token_ids, is_masked], [mlm_loss, mlm_acc]
    )

    loss = {
        'mlm_loss': lambda y_true, y_pred: y_pred,
        'mlm_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),
        #         'nsp_loss': lambda y_true, y_pred: y_pred,
        #         'nsp_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),
    }

    return bert, train_model, loss


bert, train_model, loss = build_transformer_model_with_mlm()

Opt = extend_with_weight_decay(Adam)
Opt = extend_with_gradient_accumulation(Opt)
Opt = extend_with_piecewise_linear_lr(Opt)
grad_accum_steps = 4

opt = Opt(learning_rate=learning_rate,
          exclude_from_weight_decay=['Norm', 'bias'],
          lr_schedule={int(len(train_data_generator) * epochs / grad_accum_steps * 0.1): 1.0,
                       len(train_data_generator) * epochs / grad_accum_steps: 0},
          weight_decay_rate=0.01,
          grad_accum_steps=grad_accum_steps
          )

train_model.compile(loss=loss, optimizer=opt)
# 如果传入权重，则加载。注：须在此处加载，才保证不报错。
if checkpoint_path is not None:
    bert.load_weights_from_checkpoint(checkpoint_path)

train_model.summary()

model_saved_path = './post_training/'


class ModelCheckpoint(keras.callbacks.Callback):
    """
        每20个epoch保存一次模型
    """

    def __init__(self):
        self.loss = 1e6

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] < self.loss:
            self.loss = logs['loss']

        #         print('epoch: {}, loss is : {}, lowest loss is:'.format(epoch, logs['loss'], self.loss))

        if (epoch + 1) % 10 == 0:
            bert.save_weights_as_checkpoint(model_saved_path + '-{}'.format(epoch + 1))

        token_ids, segment_ids = tokenizer.encode(u'上课时学生手机响个不停,老师一怒之下把手机摔了,家长拿发票让老师赔,大家怎么看待这种事')
        token_ids[9] = token_ids[10] = tokenizer._token_mask_id

        probs = bert.model.predict([np.array([token_ids]), np.array([segment_ids])])
        print(tokenizer.decode(probs[0, 9:11].argmax(axis=1)))


if __name__ == '__main__':
    # 保存模型
    checkpoint = ModelCheckpoint()
    # 记录日志
    csv_logger = keras.callbacks.CSVLogger('training.log')

    train_model.fit(
        train_data_generator.generator(),
        steps_per_epoch=len(train_data_generator),
        epochs=epochs,
        callbacks=[checkpoint, csv_logger],
    )
