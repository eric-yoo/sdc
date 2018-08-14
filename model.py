#
# Copyright :kissing_heart: :brain: Team. All Rights Reserved.
#

from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib

# device_lib.list_local_devices()

# Parameters
total_rows = 22000
num_epochs = 20
learning_rate = 1e-3
batch_size = 64

# TODO: Implements adapter for event columns so that event records can be passed to the network
event_dim = 512
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

def event_adapter(batch_size):
    """
    Adapter for event records.

    Arguments
      batch_size: batch size

    Returns
      event_records: event records to be returned
      query_answers: answers to query
    """

    #
    # WARN: Not working!!
    #
    filepath = os.path.join(os.getcwd(), "data_train")
    train_data_file = os.path.join(filepath, "train_data.csv")
    n = sum(1 for line in open(train_data_file)) - 1 #number of records in file (excludes header)
    skip = sorted(random.sample(range(1,n+1),n-batch_size)) #the 0-indexed header will not be included in the skip list
    event_records = pandas.read_csv(train_data_file, skiprows=skip, engine='python')

    #
    # WARN: Not working!!
    #
    query_answers = []
    query_answers.append(np.random.randint(0, answer_dim[0], batch_size))
    query_answers.append(np.random.randint(0, answer_dim[1], batch_size))
    query_answers.append(np.random.randint(0, answer_dim[2], batch_size))

    return event_records, query_answers

def test_event_adapter(batch_size):
    """
    Adapter for test event records.

    Arguments
      batch_size: batch size

    Returns
      test_event_records: evnet records to be returned
    """
    # TODO: Replace data with real data
    test_event_records = np.random.uniform(0, 1, (batch_size, event_dim))

    return test_event_records

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

# Declare loss
loss_0 = tf.losses.hinge_loss(tf.one_hot(answer_0, answer_dim[0]), logits=logits_0)
loss_1 = tf.losses.hinge_loss(tf.one_hot(answer_1, answer_dim[1]), logits=logits_1)
loss_2 = tf.losses.hinge_loss(tf.one_hot(answer_2, answer_dim[2]), logits=logits_2)
model_loss = loss_0 + loss_1 + loss_2

# Declare Optimizer
solver = tf.train.AdamOptimizer(learning_rate).minimize(model_loss)

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        sess.run(tf.global_variables_initializer())

        # Train the model
        for epoch in range(num_epochs):
            for iter in range(total_rows // batch_size):
                event_records, query_answers = event_adapter(batch_size)

                feed_dict = {
                    events: event_records,
                    answer_0: query_answers[0],
                    answer_1: query_answers[1],
                    answer_2: query_answers[2],
                }

                _, loss = sess.run([solver, model_loss], feed_dict)

                # Are we doing well?
                if iter % 1000 == 0:
                    print("Epoch: {}, Loss: {:.4}".format(epoch, loss))

                    predicted_query_answers = []
                    predicted_query_answers.append(tf.argmax(logits_0, 1))
                    predicted_query_answers.append(tf.argmax(logits_1, 1))
                    predicted_query_answers.append(tf.argmax(logits_2, 1))

                    indent = "--"
                    for i in range(3):
                        prediction_results = tf.equal(predicted_query_answers[i], query_answers[i])
                        correct_predictions = tf.reduce_sum(tf.cast(prediction_results, tf.int64))

                        total_predictions = batch_size

                        correct  = correct_predictions / total_predictions
                        accuracy = sess.run(correct, feed_dict={events: event_records})

                        print("{} accuracy for query {} : {}".format(indent, i, accuracy))

        # Let's test the model
        test_event_records = test_event_adapter(batch_size)

        # Run the forward pass
        logits_0, logits_1, logits_2 = sess.run([logits_0, logits_1, logits_2],
                                                feed_dict={events: test_event_records})

        test_query_answers = []
        test_query_answers.append(tf.argmax(logits_0, 1))
        test_query_answers.append(tf.argmax(logits_1, 1))
        test_query_answers.append(tf.argmax(logits_2, 1))

        for i in range(3):
            print("answers[{}] : {}".format(i, sess.run(test_query_answers[i])))
