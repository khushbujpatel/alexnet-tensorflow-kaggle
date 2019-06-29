"""
AlexNet with bare-metal TensorFlow Low Level API
"""
import os

import numpy as np
import tensorflow as tf


def AlexNet(features, net_data=None, keep_prob=1.0, features_extract=False):
    mu = 0
    sigma = 1e-2

    # Input ?x?x3 Output 227x227x3
    with tf.variable_scope('P0'):
        p0 = tf.image.resize_images(features, (227, 227))
        print("P0: Input %s Output %s" % (features.get_shape(), p0.get_shape()))

    # Input 227x227x3 Output 57x57x96
    with tf.variable_scope('C1'):
        if net_data:
            weight1 = tf.Variable(net_data['conv1'][0])
            bias1 = tf.Variable(net_data['conv1'][1])
        else:
            weight1 = tf.Variable(tf.truncated_normal(shape=(11, 11, 3, 96), mean=mu, stddev=sigma))
            bias1 = tf.Variable(tf.zeros(shape=(96)))
        conv1 = tf.nn.conv2d(p0, weight1, strides=(1, 4, 4, 1), padding='SAME')
        conv1 = tf.add(conv1, bias1)
        conv1 = tf.nn.relu(conv1)
        print("C1: Input %s Output %s" % (features.get_shape(), conv1.get_shape()))
        conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob)

    # Input 57x57x96 Ouput 57x57x96
    with tf.variable_scope('L1'):
        lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
        print("L1: Input %s Output %s" % (conv1.get_shape(), lrn1.get_shape()))

    # Input 57x57x96 Output 28x28x96
    with tf.variable_scope('S1'):
        pool1 = tf.nn.max_pool(lrn1, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')
        print("S1: Input %s Output %s" % (lrn1.get_shape(), pool1.get_shape()))

    # Input 28x28x96 Output 28x28x256
    with tf.variable_scope('C2'):
        if net_data:
            weight2 = tf.Variable(net_data['conv2'][0])
            bias2 = tf.Variable(net_data['conv2'][1])
        else:
            weight2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 48, 256), mean=mu, stddev=sigma))
            bias2 = tf.Variable(tf.zeros(shape=(256)))
        weight2_0, weight2_1 = tf.split(weight2, 2, 3)
        pool1_0, pool1_1 = tf.split(pool1, 2, 3)
        conv2_0 = tf.nn.conv2d(pool1_0, weight2_0, strides=(1, 1, 1, 1), padding='SAME')
        conv2_1 = tf.nn.conv2d(pool1_1, weight2_1, strides=(1, 1, 1, 1), padding='SAME')
        conv2 = tf.concat([conv2_0, conv2_1], 3)
        conv2 = tf.add(conv2, bias2)
        conv2 = tf.reshape(conv2, [-1] + conv2.get_shape().as_list()[1:])
        conv2 = tf.nn.relu(conv2)
        print("C2: Input %s Output %s" % (pool1.get_shape(), conv2.get_shape()))
        conv2 = tf.nn.dropout(conv2, keep_prob=keep_prob)

    # Input 28x28x256 Output 28x28x256
    with tf.variable_scope('L2'):
        lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
        print("L2: Input %s Output %s" % (conv2.get_shape(), lrn2.get_shape()))

    # Input 28x28x256 Output 13x13x256
    with tf.variable_scope('S2'):
        pool2 = tf.nn.max_pool(lrn2, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')
        print("S2: Input %s Output %s" % (lrn2.get_shape(), pool2.get_shape()))

    # Input 13x13x256 Output 13x13x384
    with tf.variable_scope('C3'):
        weight3 = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 384), mean=mu, stddev=sigma))
        bias3 = tf.Variable(tf.zeros(shape=(384)))
        conv3 = tf.nn.conv2d(pool2, weight3, strides=(1, 1, 1, 1), padding='SAME')
        conv3 = tf.add(conv3, bias3)
        conv3 = tf.nn.relu(conv3)
        print("C3: Input %s Output %s" % (pool2.get_shape(), conv3.get_shape()))
        conv3 = tf.nn.dropout(conv3, keep_prob=keep_prob)

    # Input 13x13x384 Output 13x13x384
    with tf.variable_scope('C4'):
        if net_data:
            weight4 = tf.Variable(net_data['conv4'][0])
            bias4 = tf.Variable(net_data['conv4'][1])
        else:
            weight4 = tf.Variable(tf.truncated_normal(shape=(3, 3, 192, 384), mean=mu, stddev=sigma))
            bias4 = tf.Variable(tf.zeros(shape=(384)))
        weight4_0, weight4_1 = tf.split(weight4, 2, 3)
        conv3_0, conv3_1 = tf.split(conv3, 2, 3)
        conv4_0 = tf.nn.conv2d(conv3_0, weight4_0, strides=(1, 1, 1, 1), padding='SAME')
        conv4_1 = tf.nn.conv2d(conv3_1, weight4_1, strides=(1, 1, 1, 1), padding='SAME')
        conv4 = tf.concat([conv4_0, conv4_1], 3)
        conv4 = tf.add(conv4, bias4)
        conv4 = tf.reshape(conv4, [-1] + conv4.get_shape().as_list()[1:])
        conv4 = tf.nn.relu(conv4)
        print("C4: Input %s Output %s" % (conv3.get_shape(), conv4.get_shape()))
        conv4 = tf.nn.dropout(conv4, keep_prob=keep_prob)

    # Input 13x13x384 Output 13x13x256
    with tf.variable_scope('C5'):
        if net_data:
            weight5 = tf.Variable(net_data['conv5'][0])
            bias5 = tf.Variable(net_data['conv5'][1])
        else:
            weight5 = tf.Variable(tf.truncated_normal(shape=(3, 3, 192, 256), mean=mu, stddev=sigma))
            bias5 = tf.Variable(tf.zeros(shape=(256)))
        weight5_0, weight5_1 = tf.split(weight5, 2, 3)
        conv4_0, conv4_1 = tf.split(conv4, 2, 3)
        conv5_0 = tf.nn.conv2d(conv4_0, weight5_0, strides=(1, 1, 1, 1), padding='SAME')
        conv5_1 = tf.nn.conv2d(conv4_1, weight5_1, strides=(1, 1, 1, 1), padding='SAME')
        conv5 = tf.concat([conv5_0, conv5_1], 3)
        conv5 = tf.add(conv5, bias5)
        conv5 = tf.reshape(conv5, [-1] + conv5.get_shape().as_list()[1:])
        conv5 = tf.nn.relu(conv5)
        print("C5: Input %s Output %s" % (conv4.get_shape(), conv5.get_shape()))
        conv5 = tf.nn.dropout(conv5, keep_prob=keep_prob)

    # Input 13x13x256 Output 6x6x256
    with tf.variable_scope('S3'):
        pool3 = tf.nn.max_pool(conv5, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')
        print("S3: Input %s Output %s" % (conv5.get_shape(), pool3.get_shape()))

    # Input 6x6x256 Output 1x9216
    with tf.variable_scope('FC5'):
        fc5 = tf.reshape(pool3,  [-1, int(np.prod(pool3.get_shape()[1:]))])

    # Input 1x9216 Output 1x4096
    with tf.variable_scope('FC6'):
        if net_data:
            weight6 = tf.Variable(net_data['fc6'][0])
            bias6 = tf.Variable(net_data['fc6'][1])
        else:
            weight6 = tf.Variable(tf.truncated_normal(shape=(9216, 4096), mean=mu, stddev=sigma))
            bias6 = tf.Variable(tf.zeros(shape=(4096)))
        print("FC6: W6 %s B6 %s" % (weight6.get_shape(), bias6.get_shape()))

        fc6 = tf.matmul(fc5, weight6)
        fc6 = tf.add(fc6, bias6)
        fc6 = tf.nn.relu(fc6)
        print("FC6: Input %s Output %s" % (pool3.get_shape(), fc6.get_shape()))
        fc6 = tf.nn.dropout(fc6, keep_prob=keep_prob)

    # Input 1x4096 Output 1x4096
    with tf.variable_scope('FC7'):
        if net_data:
            weight7 = tf.Variable(net_data['fc7'][0])
            bias7 = tf.Variable(net_data['fc7'][1])
        else:
            weight7 = tf.Variable(tf.truncated_normal(shape=(4096, 4096), mean=mu, stddev=sigma))
            bias7 = tf.Variable(tf.zeros(4096))
        fc7 = tf.matmul(fc6, weight7)
        fc7 = tf.add(fc7, bias7)
        fc7 = tf.nn.relu(fc7)
        print("FC7: Input %s Output %s" % (fc6.get_shape(), fc7.get_shape()))
        fc7 = tf.nn.dropout(fc7, keep_prob=keep_prob)

    if features_extract:
        return fc7

    # Input 1x4096 Output 1x1000
    with tf.variable_scope('FC8'):
        if net_data:
            weight8 = tf.Variable(net_data['fc8'][0])
            bias8 = tf.Variable(net_data['fc8'][1])
        else:
            weight8 = tf.Variable(tf.truncated_normal(shape=(4096, 1000), mean=mu, stddev=sigma))
            bias8 = tf.Variable(tf.zeros(1000))
        fc8 = tf.matmul(fc7, weight8)
        fc8 = tf.add(fc8, bias8)
        print("FC8: Input %s Output %s" % (fc7.get_shape(), fc8.get_shape()))

    return tf.nn.softmax(fc8)


def retrain(X_train, y_train, net_weights=None, learning_rate=0.001, epochs=10, batch_size=128, save_graph="alexnet.pb"):

    net_data = None
    nclasses = len(np.unique(y_train))
    if net_weights:
        net_data = np.load(net_weights, encoding="latin1").item()

    # prepare model to classify
    features = tf.placeholder(tf.float32, (None, None, None, 3), name='features')
    labels = tf.placeholder(tf.int64, None, name='labels')

    fc7 = AlexNet(features, net_data=net_data, features_extract=True, keep_prob=0.5)
    if net_data:
        fc7 = tf.stop_gradient(fc7)
    with tf.variable_scope('fc8'):
        weight8 = tf.Variable(tf.truncated_normal(shape=(fc7.get_shape().as_list()[-1], nclasses), mean=0, stddev=1e-2))
        bias8 = tf.Variable(tf.zeros(shape=(nclasses)))
        logits = tf.matmul(fc7, weight8)
        logits = tf.add(logits, bias8)
        print("fc8: Input %s Output %s" % (fc7.get_shape(), logits.get_shape()))

    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=832289)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss_op = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss_op, var_list=[weight8, bias8])

    preds = tf.argmax(logits, 1)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

    # saver = tf.train.Saver()
    model_dir = os.path.join(os.path.curdir, os.path.dirname(save_graph))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        total_loss = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
            accuracy, loss = sess.run([accuracy_op, loss_op], feed_dict={features: batch_x, labels: batch_y})
            total_accuracy += (accuracy * len(batch_x))
            total_loss += (loss * len(batch_x))
        return total_accuracy / num_examples, total_loss / num_examples

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training ....")
        print()
        for i in range(epochs):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                sess.run(training_op, feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})

            training_accuracy, training_loss = evaluate(X_train, y_train)
            validation_accuracy, validation_loss = evaluate(X_valid, y_valid)

            print("EPOCH {} ...".format(i+1))
            print("Training Accuracy = {:.3f} Loss = {:.3f}".format(training_accuracy, training_loss))
            print("Validation Accuracy = {:.3f} Loss = {:.3f}".format(validation_accuracy, validation_loss))
            print()

        # saver.save(sess, os.path.join(model_dir, "alexnet.ckpt"))

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ["fc8/Add"])

        with tf.gfile.GFile(os.path.join(model_dir, save_graph), "wb") as f:
            f.write(output_graph_def.SerializeToString())

def inference(fname, model, labels):
    """
    Args
        fname (str) - image filepath (*.jpg)
        model (str) - model filepath (*.pb)
        labels(str) - labels filepath (*.txt)

    Return
        predicted_label (str) - prediction result
    """
    import cv2
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    image = cv2.imread(fname)
    assert image is not None, "Failed to open [%s]" % (fname)

    image = image - np.mean(image)

    input_tensor = graph.get_tensor_by_name("import/features:0")
    output_tensor = graph.get_tensor_by_name("import/fc8/Add:0")

    with tf.Session(graph=graph) as sess:
        predictions = sess.run(output_tensor, feed_dict={input_tensor: np.expand_dims(image, 0)})
    results = np.squeeze(predictions)

    top_k = results.argsort()[-5:][::-1]
    class_names = [l.strip() for l in tf.gfile.GFile(labels).readlines()]
    for i in top_k:
        print(class_names[i], results[i])

    return results[top_k[0]]

if __name__ == "__main__":

    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_p", default="train.p", help="path to train.p")
    parser.add_argument("--test_p", default="test.p", help="path to test.p")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for training")
    parser.add_argument("--epochs", type=int, default=2, help="number of training iterations")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--net_weights", default=None, help="pretrained weights bvlc-alexnet.npy")
    parser.add_argument("--train", action="store_true", help="train/retrain model")
    parser.add_argument("--save_graph", default="alexnet.pb", help="saves graph as *.pb while training or loads *.pb for inference")
    parser.add_argument("--labels", default="labels.txt", help="labels file")
    parser.add_argument("--infer", default=None, help="perform inference for file")
    args = parser.parse_args()

    print(args)

    trainset = pickle.load(open(args.train_p, "rb"))
    testset = pickle.load(open(args.test_p, "rb"))

    X_train, y_train = trainset['features'], trainset['labels']
    X_test, y_test = testset['features'], testset['labels']

    if args.train:
        retrain(X_train, y_train, net_weights=args.net_weights, epochs=args.epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate, save_graph=args.save_graph)

    if args.infer:
        inference(args.infer, args.save_graph, args.labels)
