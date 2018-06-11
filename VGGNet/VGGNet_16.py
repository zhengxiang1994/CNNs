import tensorflow as tf
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import cv2


def conv2d(x, W, b, strides=1):
    # conv2d wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # max pooling wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")


def conv_net(x, weights, biases, dropout):
    # reshape input picture
    x = tf.reshape(x, shape=[-1, 128, 128, 1])  # 128x128x3

    # conv layer
    conv1 = conv2d(x, weights["wc1"], biases["bc1"])
    conv2 = conv2d(conv1, weights["wc2"], biases["bc2"])
    # max pooling
    pool1 = maxpool2d(conv2, k=2)   # 64x64x64

    # conv layer
    conv3 = conv2d(pool1, weights["wc3"], biases["bc3"])
    conv4 = conv2d(conv3, weights["wc4"], biases["bc4"])
    # max pooling
    pool2 = maxpool2d(conv4, k=2)   # 32x32x128

    # conv layer
    conv5 = conv2d(pool2, weights["wc5"], biases["bc5"])
    conv6 = conv2d(conv5, weights["wc6"], biases["bc6"])
    conv7 = conv2d(conv6, weights["wc7"], biases["bc7"])
    # max pooling
    pool3 = maxpool2d(conv7, k=2)   # 16x16x256

    # conv layer
    conv8 = conv2d(pool3, weights["wc8"], biases["bc8"])
    conv9 = conv2d(conv8, weights["wc9"], biases["bc9"])
    conv10 = conv2d(conv9, weights["wc10"], biases["bc10"])
    # max pooling
    pool4 = maxpool2d(conv10, k=2)  # 8x8x512

    # conv layer
    conv11 = conv2d(pool4, weights["wc11"], biases["bc11"])
    conv12 = conv2d(conv11, weights["wc12"], biases["bc12"])
    conv13 = conv2d(conv12, weights["wc13"], biases["bc13"])
    # max pooling
    pool5 = maxpool2d(conv13, k=2)  # 4x4x512

    # fully connected layer
    # reshape conv2d output to fit fully connected layer input
    fc1 = tf.reshape(pool5, [-1, weights["wd1"].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, weights["wd1"]), biases["bd1"]))
    # apply dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # fully connected layer
    fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, weights["wd2"]), biases["bd2"]))
    # apply dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # prediction
    out = tf.add(tf.matmul(fc2, weights["out"]), biases["out"])
    return out


def next_batch(num, data, labels):
    # Return a total of `num` random samples and labels.
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def img_resize(imgs):
    imgs_128 = []
    # img resize
    for img in imgs:
        img = img.reshape(28, 28)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        imgs_128.append(img)
    imgs_128 = np.array(imgs_128)
    return imgs_128 / 255.


weights = {
    # 3x3 conv, 1 input, 64 output
    "wc1": tf.Variable(tf.random_normal([3, 3, 1, 64])),
    "wc2": tf.Variable(tf.random_normal([3, 3, 64, 64])),

    "wc3": tf.Variable(tf.random_normal([3, 3, 64, 128])),
    "wc4": tf.Variable(tf.random_normal([3, 3, 128, 128])),

    "wc5": tf.Variable(tf.random_normal([3, 3, 128, 256])),
    "wc6": tf.Variable(tf.random_normal([3, 3, 256, 256])),
    "wc7": tf.Variable(tf.random_normal([3, 3, 256, 256])),

    "wc8": tf.Variable(tf.random_normal([3, 3, 256, 512])),
    "wc9": tf.Variable(tf.random_normal([3, 3, 512, 512])),
    "wc10": tf.Variable(tf.random_normal([3, 3, 512, 512])),

    "wc11": tf.Variable(tf.random_normal([3, 3, 512, 512])),
    "wc12": tf.Variable(tf.random_normal([3, 3, 512, 512])),
    "wc13": tf.Variable(tf.random_normal([3, 3, 512, 512])),

    # fully connected, 4x4x512 input, 1024 output
    "wd1": tf.Variable(tf.random_normal([4*4*512, 1024])),
    "wd2": tf.Variable(tf.random_normal([1024, 1024])),

    # prediction, 1024 input, 10 output
    "out": tf.Variable(tf.random_normal([1024, 10]))
}

biases = {
    "bc1": tf.Variable(tf.random_normal([64])),
    "bc2": tf.Variable(tf.random_normal([64])),
    "bc3": tf.Variable(tf.random_normal([128])),
    "bc4": tf.Variable(tf.random_normal([128])),
    "bc5": tf.Variable(tf.random_normal([256])),
    "bc6": tf.Variable(tf.random_normal([256])),
    "bc7": tf.Variable(tf.random_normal([256])),
    "bc8": tf.Variable(tf.random_normal([512])),
    "bc9": tf.Variable(tf.random_normal([512])),
    "bc10": tf.Variable(tf.random_normal([512])),
    "bc11": tf.Variable(tf.random_normal([512])),
    "bc12": tf.Variable(tf.random_normal([512])),
    "bc13": tf.Variable(tf.random_normal([512])),
    "bd1": tf.Variable(tf.random_normal([1024])),
    "bd2": tf.Variable(tf.random_normal([1024])),
    "out": tf.Variable(tf.random_normal([10])),
}


if __name__ == "__main__":
    # data preparation
    train_imgs_ori = sio.loadmat("../mnist_data/train_images.mat")["train_images"]
    test_imgs_ori = sio.loadmat(r"../mnist_data/test_images.mat")["test_images"]
    train_imgs = img_resize(train_imgs_ori)
    train_labels = sio.loadmat(r"../mnist_data/train_labels.mat")["train_labels"]
    test_imgs = img_resize(test_imgs_ori)
    test_labels = sio.loadmat(r"../mnist_data/test_labels.mat")["test_labels"]
    enc = OneHotEncoder()
    train_labels_enc = enc.fit_transform(train_labels).toarray()
    test_labels_enc = enc.fit_transform(test_labels).toarray()

    # img = test_imgs[0]
    # cv2.imshow("", img)
    # cv2.waitKey()

    keep_prob = 1.
    max_iters = 200
    x = tf.placeholder(tf.float32, [None, 128, 128])
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
            if i % 1 == 0:
                # print(sess.run(pred, feed_dict={x: x_batch[0]}))
                print(sess.run([accuracy, cost], feed_dict={x: x_batch[0], y_: x_batch[1]}))
            sess.run(optimizer, feed_dict={x: x_batch[0], y_: x_batch[1]})
        print("training ends")
        print("test accuracy:", sess.run(accuracy, feed_dict={x: test_imgs, y_: test_labels_enc}))










