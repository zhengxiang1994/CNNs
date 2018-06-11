import tensorflow as tf
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import OneHotEncoder

weights = {
    # 3x3 conv, 1 input, 64 output
    "wc1": tf.Variable(tf.random_normal([3, 3, 1, 64])),
    "wc2": tf.Variable(tf.random_normal([3, 3, 64, 64])),

    "wc3": tf.Variable(tf.random_normal([3, 3, 64, 128])),
    "wc4": tf.Variable(tf.random_normal([3, 3, 128, 128])),

    "wc5": tf.Variable(tf.random_normal([3, 3, 128, 256])),
    "wc6": tf.Variable(tf.random_normal([3, 3, 256, 256])),
    "wc7": tf.Variable(tf.random_normal([3, 3, 256, 256])),

    "wc8": tf.Variable(tf.random_normal([3, 3, 256, 256])),

    # fully connected, 4x4x512 input, 1024 output
    "wd1": tf.Variable(tf.random_normal([7*7*256, 1024])),
    "wd2": tf.Variable(tf.random_normal([1024, 1024])),

    # prediction, 1024 input, 10 output
    "wd3": tf.Variable(tf.random_normal([1024, 10]))
}

biases = {
    "bc1": tf.Variable(tf.random_normal([64])),
    "bc2": tf.Variable(tf.random_normal([64])),
    "bc3": tf.Variable(tf.random_normal([128])),
    "bc4": tf.Variable(tf.random_normal([128])),
    "bc5": tf.Variable(tf.random_normal([256])),
    "bc6": tf.Variable(tf.random_normal([256])),
    "bc7": tf.Variable(tf.random_normal([256])),
    "bc8": tf.Variable(tf.random_normal([256])),
    "bd1": tf.Variable(tf.random_normal([1024])),
    "bd2": tf.Variable(tf.random_normal([1024])),
    "bd3": tf.Variable(tf.random_normal([10])),
}


def conv2d(x, W, b, strides=1):
    # conv2d wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return x


def maxpool2d(x, k=2):
    # max pooling wrapper
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, k, k, 1], padding="SAME")


def conv_net(x, weights, biases, dropout):
    # reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])    # 28x28x3

    # layer1
    h_conv1 = tf.nn.relu(conv2d(x, weights["wc1"], biases["bc1"]))  # 28x28x64
    # print(h_conv1.get_shape().as_list())

    # layer2
    h_conv2 = tf.nn.relu(conv2d(h_conv1, weights["wc2"], biases["bc2"]))    # 28x28x64
    h_pool2 = maxpool2d(h_conv2, k=2)   # 14x14x64

    # layer3
    h_conv3 = tf.nn.relu(conv2d(h_pool2, weights["wc3"], biases["bc3"]))    # 14x14x128

    # layer4
    h_conv4 = tf.nn.relu(conv2d(h_conv3, weights["wc4"], biases["bc4"]))    # 14x14x128
    h_pool4 = maxpool2d(h_conv4, k=2)   # 7x7x128

    # layer5
    h_conv5 = tf.nn.relu(conv2d(h_pool4, weights["wc5"], biases["bc5"]))    # 7x7x256

    # layer6
    h_conv6 = tf.nn.relu(conv2d(h_conv5, weights["wc6"], biases["bc6"]))    # 7x7x256

    # layer7
    h_conv7 = tf.nn.relu(conv2d(h_conv6, weights["wc7"], biases["bc7"]))    # 7x7x256

    # layer8
    h_conv8 = tf.nn.relu(conv2d(h_conv7, weights["wc8"], biases["bc8"]))    # 7x7x256
    h_pool8 = maxpool2d(h_conv8, k=1)   # 7x7x256

    # layer9
    h_pool8_flat = tf.reshape(h_pool8, [-1, 7*7*256])
    h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool8_flat, weights["wd1"]), biases["bd1"]))
    h_fc1_drop = tf.nn.dropout(h_fc1, dropout)

    # layer10
    h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fc1_drop, weights["wd2"]), biases["bd2"]))
    h_fc2_drop = tf.nn.dropout(h_fc2, dropout)

    # layer11
    h_fc3 = tf.add(tf.matmul(h_fc2_drop, weights["wd3"]), biases["bd3"])
    return h_fc3


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

    keep_prob = 1.
    max_iters = 200
    x = tf.placeholder(tf.float32, [None, 28*28])
    y_ = tf.placeholder(tf.float32, [None, 10])
    pred = conv_net(x, weights, biases, keep_prob)

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





