# LeNet with batch normalization
# model save and restore
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import os


def next_batch(num, data, labels):
    # Return a total of `num` random samples and labels.
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    bneepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bneepsilon)
    return Ybn, update_moving_averages


def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()


def compativle_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
    return noiseshape


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bais_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def con2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


if __name__ == "__main__":
    # data preparation
    train_imgs = sio.loadmat("../mnist_data/train_images.mat")["train_images"]
    train_labels = sio.loadmat(r"../mnist_data/train_labels.mat")["train_labels"]
    test_imgs = sio.loadmat(r"../mnist_data/test_images.mat")["test_images"]
    test_labels = sio.loadmat(r"../mnist_data/test_labels.mat")["test_labels"]
    enc = OneHotEncoder()
    train_labels_enc = enc.fit_transform(train_labels).toarray()
    test_labels_enc = enc.fit_transform(test_labels).toarray()
    # print(train_labels_enc)

    # construct network
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bais_variable([32])
    h_conv1 = tf.nn.relu(con2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bais_variable([64])
    h_conv2 = tf.nn.relu(con2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    w_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bais_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    w_fc2 = weight_variable([1024, 512])
    b_fc2 = bais_variable([512])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

    w_fc3 = weight_variable([512, 10])
    b_fc3 = bais_variable([10])
    # y_conv = tf.nn.softmax(tf.matmul(h_fc2, w_fc3) + b_fc3)
    y_conv = tf.matmul(h_fc2, w_fc3) + b_fc3

    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(200):
            x_batch = next_batch(128, train_imgs, train_labels_enc)
            if i % 10 == 0:
                print(sess.run([accuracy, cross_entropy], feed_dict={x: x_batch[0], y_: x_batch[1]}))
            sess.run(train_step, feed_dict={x: x_batch[0], y_: x_batch[1]})

        print("test accuracy:", sess.run([accuracy], feed_dict={x: test_imgs, y_: test_labels_enc}))

        if not os.path.exists("models/"):
            os.makedirs("models/")
        saver.save(sess, "models/model.ckpt")
        print("model stored!")

    # with tf.Session() as sess:
    #     saver.restore(sess, "models/model.ckpt")
    #     print("[+] Test accuracy is {0}".format(sess.run(accuracy, feed_dict={x: test_imgs, y_: test_labels_enc})))



