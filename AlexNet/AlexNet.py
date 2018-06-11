import tensorflow as tf
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import OneHotEncoder

weights = {
    # 3x3 conv, 1 input, 64 output
    "wc1": tf.Variable(tf.truncated_normal([3, 3, 1, 64])),
    "wc2": tf.Variable(tf.truncated_normal([3, 3, 64, 128])),

    "wc3": tf.Variable(tf.truncated_normal([3, 3, 128, 256])),
    "wc4": tf.Variable(tf.truncated_normal([3, 3, 256, 256])),

    "wc5": tf.Variable(tf.truncated_normal([3, 3, 256, 128])),

    # fully connected, 4x4x512 input, 1024 output
    "wd1": tf.Variable(tf.truncated_normal([4*4*128, 1024])),
    "wd2": tf.Variable(tf.random_normal([1024, 1024])),

    # prediction, 1024 input, 10 output
    "wd3": tf.Variable(tf.random_normal([1024, 10]))
}

biases = {
    "bc1": tf.Variable(tf.constant(0., shape=[64])),
    "bc2": tf.Variable(tf.constant(0., shape=[128])),
    "bc3": tf.Variable(tf.constant(0., shape=[256])),
    "bc4": tf.Variable(tf.constant(0., shape=[256])),
    "bc5": tf.Variable(tf.constant(0., shape=[128])),
    "bd1": tf.Variable(tf.constant(0., shape=[1024])),
    "bd2": tf.Variable(tf.constant(0., shape=[1024])),
    "bd3": tf.Variable(tf.constant(0., shape=[10])),
}


def conv2d(x, W, b, strides=1):
    # conv2d wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return x


def maxpool2d(x, k=2):
    # max pooling wrapper
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, k, k, 1], padding="SAME")


def alex_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # layer1
    conv1 = tf.nn.relu(conv2d(x, weights["wc1"], biases["bc1"]))     # 28x28x64

    # layer LRN, effect is not significant and result in speed slow, so abandon it
    # lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75)

    pool1 = maxpool2d(conv1, k=2)   # 14x14x64

    # layer2
    conv2 = tf.nn.relu(conv2d(pool1, weights["wc2"], biases["bc2"]))     # 14x14x128
    pool2 = maxpool2d(conv2, k=2)   # 7x7x128

    # layer3
    conv3 = tf.nn.relu(conv2d(pool2, weights["wc3"], biases["bc3"]))     # 7x7x256

    # layer4
    conv4 = tf.nn.relu(conv2d(conv3, weights["wc4"], biases["bc4"]))    # 7x7x256

    # layer5
    conv5 = tf.nn.relu(conv2d(conv4, weights["wc5"], biases["bc5"]))    # 7x7x128
    pool5 = maxpool2d(conv5, k=2)   # 4x4x128

    # layer6
    pool5_flat = tf.reshape(pool5, [-1, weights["wd1"].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.add(tf.matmul(pool5_flat, weights["wd1"]), biases["bd1"]))
    fc1_drop = tf.nn.dropout(fc1, dropout)

    # layer7
    fc2 = tf.nn.relu(tf.add(tf.matmul(fc1_drop, weights["wd2"]), biases["bd2"]))
    fc2_drop = tf.nn.dropout(fc2, dropout)

    # layer8
    fc3 = tf.add(tf.matmul(fc2_drop, weights["wd3"]), biases["bd3"])
    return fc3


def next_batch(num, data, labels):
    # Return a total of `num` random samples and labels.
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


if __name__ == "__main__":
    # data preparation
    train_imgs = sio.loadmat("../mnist_data/train_images.mat")["train_images"]
    train_labels = sio.loadmat(r"../mnist_data/train_labels.mat")["train_labels"]
    test_imgs = sio.loadmat(r"../mnist_data/test_images.mat")["test_images"]
    test_labels = sio.loadmat(r"../mnist_data/test_labels.mat")["test_labels"]
    enc = OneHotEncoder()
    train_labels_enc = enc.fit_transform(train_labels).toarray()
    test_labels_enc = enc.fit_transform(test_labels).toarray()

    keep_prob = 0.75
    max_iters = 2000
    x = tf.placeholder(tf.float32, [None, 28*28])

    y_ = tf.placeholder(tf.float32, [None, 10])
    pred = alex_net(x, weights, biases, keep_prob)

    # define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    # evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("training starts")
        for i in range(max_iters):
            x_batch = next_batch(64, train_imgs, train_labels_enc)
            if i % 10 == 0:
                # print(sess.run(pred, feed_dict={x: x_batch[0]}))
                print("step {},".format(i), sess.run([accuracy, cost], feed_dict={x: x_batch[0], y_: x_batch[1]}))
            sess.run(optimizer, feed_dict={x: x_batch[0], y_: x_batch[1]})
        print("training ends")
        print("test accuracy:", sess.run(accuracy, feed_dict={x: test_imgs[:1000], y_: test_labels_enc[:1000]}))





