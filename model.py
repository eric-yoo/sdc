#
# Copyright :kissing_heart: :brain: Team. All Rights Reserved.
#

from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
import pandas
import random
from tensorflow.python.client import device_lib
from sklearn.preprocessing import LabelEncoder

# load csv and encode as df

encode_index=['주야(6시,18시 기준)', '오전/오후', '요일', '요일_주야','사망자 피해수준(하 0~2 / 중 3~4 / 상 5~)','부상자 피해수준(하 0~5 / 중 6~18 / 상 19~)',  '발생지시도', '발생지시군구', '발생지시도군구', '사고유형_대분류', '사고유형_중분류', '사고유형',
'사망자피해수준_사고유형중분류', '부상자피해수준_사고유형중분류', '발생지_사고유형대분류', '발생지_사고유형중분류',
'법규위반_대분류', '법규위반', '도로형태_대분류', '도로형태', '도로형태(전체)', '도로형태*법규위반',
'당사자종별_1당_대분류', '당사자종별_1당', '당사자종별_2당_대분류', '당사자종별_2당',  '가해자_피해자', '가해자_도로형태', '가해자_법규위반']

filepath = os.path.join(os.getcwd(), "data_train")
train_data_file = os.path.join(filepath, "train.csv")
df = pandas.read_csv(train_data_file, encoding="utf-8", engine='python')
# df.dtypes.index
event_records = df.values

# print(event_records)
encoder_list=[]
for i in encode_index:
    le = LabelEncoder()
    df[i]= le.fit_transform(df[i])
    encoder_list.append(le)

# device_lib.list_local_devices()

# TODO: Better command line option parsing?
device_name = sys.argv[0]
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

# Parameters
num_epochs = 80
learning_rate = 1e-3
batch_size = 64

# TODO: Implements adapter for event columns so that event records can be passed to the network
event_dim = 46
embedding_dim = 128 # input column dimension
encoder_net_hidden_layers = [512, 512, 256] # hidden layer dimensions for base network
latent_dim = 20 # latent variable dimension

# The number of possible answers for each query
answer_dim = [10, 20, 30]

# Placeholders
events = tf.placeholder(tf.float32, shape=[None, event_dim])
answer_0 = tf.placeholder(tf.int64, [None])
answer_1 = tf.placeholder(tf.int64, [None])
answer_2 = tf.placeholder(tf.int64, [None])
answer_3 = tf.placeholder(tf.int64, [None])
answer_4 = tf.placeholder(tf.int64, [None])
answer_5 = tf.placeholder(tf.int64, [None])
answer_6 = tf.placeholder(tf.int64, [None])
answer_7 = tf.placeholder(tf.int64, [None])
answer_8 = tf.placeholder(tf.int64, [None])
answer_9 = tf.placeholder(tf.int64, [None])
answer_10 = tf.placeholder(tf.int64, [None])
answer_11 = tf.placeholder(tf.int64, [None])
answer_12 = tf.placeholder(tf.int64, [None])
answer_13 = tf.placeholder(tf.int64, [None])
answer_14 = tf.placeholder(tf.int64, [None])
answer_15 = tf.placeholder(tf.int64, [None])

def event_adapter(batch_size):
    """
    Adapter for event records.

    Arguments
      batch_size: batch size

    Returns
      event_records: event records to be returned
      query_answers: answers to query
    """

    sample = df.sample(n=batch_size, axis=0)

    event_records = sample.values

    query_index=['주야(6시,18시 기준)', '요일', '사망자수',  '사상자수', '중상자수', '경상자수', '부상신고자수', '발생지시도', '발생지시군구','사고유형_대분류', '사고유형_중분류','법규위반', '도로형태_대분류', '도로형태',  '당사자종별_1당_대분류',  '당사자종별_2당_대분류', ]

    query_answers = sample[query_index].values

    return event_records, query_answers

def encoder_net(events_embedded):
    """
    Encoder network model for event.

    Arguments
      events: Event tensors

    Returns
      latent_variabes: Latent variables as an output of the encoder network
    """
    # Hidden layers
    h1 = tf.layers.dense(
        inputs=events_embedded,
        units=encoder_net_hidden_layers[0],
        activation=tf.nn.relu
    )

    h2 = tf.layers.dense(
        inputs=h1,
        units=encoder_net_hidden_layers[1],
        activation=tf.nn.relu
    )

    h3 = tf.layers.dense(
        inputs=h2,
        units=encoder_net_hidden_layers[2],
        activation=tf.nn.relu
    )

    # output layer
    latent_variables = tf.layers.dense(
        inputs=h3, units=latent_dim, activation=None
    )

    return latent_variables


def query_net(latent_variables, hidden_layers, output_dim):
    """
    Query network that takes latent variables and outputs answers for the specified query.

    Arguments
      latent_variables: Latent variables; output of encoder network
      hidden_layers: :ist of two integers; dimensions  for hidden layers
      output_dim: Query specific output dimension

    Returns
      logits: Output logits
    """
    h1 = tf.layers.dense(
        inputs=latent_variables,
        units=hidden_layers[0],
        activation=tf.nn.relu
    )

    h2 = tf.layers.dense(
        inputs=h1,
        units=hidden_layers[1],
        activation=tf.nn.relu
    )

    logits = tf.layers.dense(
        inputs=h2,
        units=output_dim,
        activation=None
    )

    return logits


# Declare model
embeddings = tf.Variable(tf.random_uniform([event_dim, embedding_dim], -1.0, 1.0))
events_embedded = tf.matmul(events, embeddings)
latent = encoder_net(events_embedded)
logits_0 = query_net(latent, [64, 64], answer_dim[0])
logits_1 = query_net(latent, [64, 64], answer_dim[1])
logits_2 = query_net(latent, [64, 64], answer_dim[2])
logits_3 = query_net(latent, [64, 64], answer_dim[3])
logits_4 = query_net(latent, [64, 64], answer_dim[4])
logits_5 = query_net(latent, [64, 64], answer_dim[5])
logits_6 = query_net(latent, [64, 64], answer_dim[6])
logits_7 = query_net(latent, [64, 64], answer_dim[7])
logits_8 = query_net(latent, [64, 64], answer_dim[8])
logits_9 = query_net(latent, [64, 64], answer_dim[9])
logits_10 = query_net(latent, [64, 64], answer_dim[10])
logits_11 = query_net(latent, [64, 64], answer_dim[11])
logits_12 = query_net(latent, [64, 64], answer_dim[12])
logits_13 = query_net(latent, [64, 64], answer_dim[13])
logits_14 = query_net(latent, [64, 64], answer_dim[14])
logits_15 = query_net(latent, [64, 64], answer_dim[15])

# Declare loss
loss_0 = tf.losses.hinge_loss(tf.one_hot(answer_0, answer_dim[0]), logits=logits_0)
loss_1 = tf.losses.hinge_loss(tf.one_hot(answer_1, answer_dim[1]), logits=logits_1)
loss_2 = tf.losses.hinge_loss(tf.one_hot(answer_2, answer_dim[2]), logits=logits_2)
loss_3 = tf.losses.hinge_loss(tf.one_hot(answer_3, answer_dim[3]), logits=logits_3)
loss_4 = tf.losses.hinge_loss(tf.one_hot(answer_4, answer_dim[4]), logits=logits_4)
loss_5 = tf.losses.hinge_loss(tf.one_hot(answer_5, answer_dim[5]), logits=logits_5)
loss_6 = tf.losses.hinge_loss(tf.one_hot(answer_6, answer_dim[6]), logits=logits_6)
loss_7 = tf.losses.hinge_loss(tf.one_hot(answer_7, answer_dim[7]), logits=logits_7)
loss_8 = tf.losses.hinge_loss(tf.one_hot(answer_8, answer_dim[8]), logits=logits_8)
loss_9 = tf.losses.hinge_loss(tf.one_hot(answer_9, answer_dim[9]), logits=logits_9)
loss_10 = tf.losses.hinge_loss(tf.one_hot(answer_10, answer_dim[10]), logits=logits_10)
loss_11 = tf.losses.hinge_loss(tf.one_hot(answer_11, answer_dim[11]), logits=logits_11)
loss_12 = tf.losses.hinge_loss(tf.one_hot(answer_12, answer_dim[12]), logits=logits_12)
loss_13 = tf.losses.hinge_loss(tf.one_hot(answer_13, answer_dim[13]), logits=logits_13)
loss_14 = tf.losses.hinge_loss(tf.one_hot(answer_14, answer_dim[14]), logits=logits_14)
loss_15 = tf.losses.hinge_loss(tf.one_hot(answer_15, answer_dim[15]), logits=logits_15)
model_loss = loss_0 + loss_1 + loss_2 +loss_3 + loss_4 + loss_5 + loss_6 + loss_7 + loss_8 +loss_9 + loss_10+loss_11 + loss_12 + loss_13 + loss_14 + loss_15

# Declare Optimizer
solver = tf.train.AdamOptimizer(learning_rate).minimize(model_loss)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.device(device_name):
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            events_records, query_answers = event_adapter(batch_size)

            feed_dict = {
                events: events_records,
                answer_0: query_answers[0],
                answer_1: query_answers[1],
                answer_2: query_answers[2],
                answer_3: query_answers[3],
                answer_4: query_answers[4],
                answer_5: query_answers[5],
                answer_6: query_answers[6],
                answer_7: query_answers[7],
                answer_8: query_answers[8],
                answer_9: query_answers[9],
                answer_10: query_answers[10],
                answer_11: query_answers[11],
                answer_12: query_answers[12],
                answer_13: query_answers[13],
                answer_14: query_answers[14],
                answer_15: query_answers[15],

            }

            _, loss = sess.run([solver, model_loss], feed_dict)

            if epoch % 5 == 0:
                print("Epoch: {}, Loss: {:.4}".format(epoch, loss))
