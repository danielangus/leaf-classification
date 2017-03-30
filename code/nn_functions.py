import numpy as np
import tensorflow as tf

from time import time
from tqdm import tqdm

from feature_functions import printer
from parameters import NN_PARAM

def one_hot_encode(data):
    # helper function to one-hot encode labels
    num_classes = NN_PARAM['output_classes']
    newdata = (np.arange(num_classes) == data[:, None]).astype(np.float32)
    return newdata

def accuracy(predictions, labels):
    # accuracy function for neural network
    # predictions and labels are one-hot encoded
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def run_tf_nn(X_train, y_train, X_test, y_test):
    # this function trains and runs the neural network
    printer("Running tensorflow neural network...")

    results = dict()
    results['name'] = 'NeuralNetwork'

    # Preprocess data for the neural network
    valid_size = int(len(X_train) * 0.2)
    X_valid = X_train.as_matrix()[0:valid_size].astype(np.float32)
    y_valid = one_hot_encode(y_train[0:valid_size])
    X_train = X_train.as_matrix()[valid_size:].astype(np.float32)
    y_train = one_hot_encode(y_train[valid_size:])
    X_test = X_test.as_matrix().astype(np.float32)
    y_test = one_hot_encode(y_test)

    # Tensor dimensions are needed to declare variable shapes
    tensor_length, tensor_width = X_train.shape[0], X_train.shape[1]

    printer("Building graph...")
    graph = tf.Graph()

    with graph.as_default():
        # -----Graph input data-----

        tf_train_dataset = tf.placeholder(name='train_data', dtype=tf.float32,
                                          shape=[None, tensor_width])
        tf_train_labels = tf.placeholder(name='train_labels', dtype=tf.float32,
                                         shape=[None, NN_PARAM['output_classes']])
        tf_valid_dataset = tf.constant(name='valid_data', value=X_valid)
        tf_test_dataset = tf.constant(name='test_data', value=X_test)

        # -----Graph variables-----

        # Fully connected layer 1:
        # [batch_size, tensor_width] -> [tensor_width, layer1_neurons]
        fc1_w = tf.get_variable(name="fc1_w",
                                shape=[tensor_width, NN_PARAM['layer1_neurons']],
                                initializer=tf.contrib.layers.xavier_initializer())
        fc1_b = tf.Variable(name='fc1_b',
                            initial_value=tf.zeros(NN_PARAM['layer1_neurons']))

        # Fully connected layer 2:
        # [tensor_width, layer1_neurons] -> [layer1_neurons, layer2_neurons]
        fc2_w = tf.get_variable(name='fc2_w',
                                shape=[NN_PARAM['layer1_neurons'], NN_PARAM['layer2_neurons']],
                                initializer=tf.contrib.layers.xavier_initializer())
        fc2_b = tf.Variable(name='fc2_b',
                            initial_value=tf.zeros(NN_PARAM['layer2_neurons']))

        # Fully connected layer 3:
        # [layer1_neurons, layer2_neurons] -> [layer2_neurons, layer3_neurons]
        fc3_w = tf.get_variable(name='fc3_w',
                                shape=[NN_PARAM['layer2_neurons'], NN_PARAM['layer3_neurons']],
                                initializer=tf.contrib.layers.xavier_initializer())
        fc3_b = tf.Variable(name='fc3_b',
                            initial_value=tf.zeros(NN_PARAM['layer3_neurons']))

        # Output layer:
        # [layer2_neurons, layer3_neurons] -> [layer3_neurons, output_classes]
        ol_w = tf.get_variable(name='ol_w',
                               shape=[NN_PARAM['layer3_neurons'], NN_PARAM['output_classes']],
                               initializer=tf.contrib.layers.xavier_initializer())
        ol_b = tf.Variable(name='ol_b',
                           initial_value=tf.zeros(NN_PARAM['output_classes']))

        # -----Model-----
        def model(data, training):
            # Layer 1
            fc1 = tf.nn.relu(tf.matmul(data, fc1_w) + fc1_b)
            if training and NN_PARAM['dropout']:
                fc1 = tf.nn.dropout(fc1, keep_prob=NN_PARAM['keep_prob'])

            # Layer 2
            fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)
            if training and NN_PARAM['dropout']:
                fc2 = tf.nn.dropout(fc2, keep_prob=NN_PARAM['keep_prob'])

            # Layer 3
            fc3 = tf.nn.relu(tf.matmul(fc2, fc3_w) + fc3_b)
            if training and NN_PARAM['dropout']:
                fc3 = tf.nn.dropout(fc3, keep_prob=NN_PARAM['keep_prob'])

            # Output layer
            return tf.matmul(fc3, ol_w) + ol_b

        # -----Computation-----
        logits = model(tf_train_dataset, training=True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        if NN_PARAM['l2_reg']:
            for weight_matrix in [fc1_w, fc2_w, fc3_w, ol_w]:
                loss += NN_PARAM['beta'] * tf.nn.l2_loss(weight_matrix)

        # -----Optimizer-----
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate=NN_PARAM['learning_rate'],
                                                   global_step=global_step,
                                                   decay_rate=NN_PARAM['decay_rate'],
                                                   decay_steps=NN_PARAM['decay_steps'],
                                                   staircase=NN_PARAM['staircase'])
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # -----Predicitions-----
        train_prediction = tf.nn.softmax(model(tf_train_dataset, training=False))
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset, training=False))
        test_prediction = tf.nn.softmax(model(tf_test_dataset, training=False))

        # -----Saver-----
        saver = tf.train.Saver()

    printer("Graph complete!")
    printer("Running session...")

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        step_losses = list()
        start = time()

        for step in tqdm(range(NN_PARAM['num_steps'])):
            # the offset steps through the training dataset
            offset = (step * NN_PARAM['batch_size']) % (tensor_length - NN_PARAM['batch_size'])

            # generate batches
            batch_data = X_train[offset:(offset + NN_PARAM['batch_size']), :]
            batch_labels = y_train[offset:(offset + NN_PARAM['batch_size'])]

            # feed the variables to session
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)

            # Every once in a while, save training data
            # To output this data, set verbose=True
            if (step % 2000 == 0):
                step_losses.append('Minibatch loss at step {}: {}'.format(step, l))
                step_losses.append('Minibatch accuracy: {:.3f}'.format(accuracy(predictions, batch_labels)))
                step_losses.append('Validation accuracy: {:.3f}'.format(accuracy(
                    valid_prediction.eval(), y_valid)))

        # -----Training Results-----
        results['train_time'] = time() - start
        printer("Training summary:")
        printer("Total time: {:.2f} seconds".format(results['train_time']))

        if NN_PARAM['verbose']:
            for statement in step_losses:
                printer('\t{}'.format(statement))

        # -----Format Results-----
        train_feed_dict = {tf_train_dataset: X_train, tf_train_labels: y_train}
        train_pred = session.run(train_prediction, feed_dict=train_feed_dict)
        results['train_acc'] = accuracy(train_pred, y_train)
        results['test_acc'] = accuracy(test_prediction.eval(), y_test)
        printer('Test accuracy: {:.3f}'.format(results['test_acc']))

        save_path = saver.save(session, "./tf_check/tf_ckpt_{}_points".format(len(X_train)))
        printer('Model saved to {}\n'.format(save_path))

    return results
