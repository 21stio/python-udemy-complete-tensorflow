import numpy as np
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True, )

tf.reset_default_graph()

n_in = 784
n_hid1 = n_in // 4
n_hid2 = n_hid1 // 4
n_hid3 = n_hid1
n_out = n_in

learning_rate = 0.001
actf = tf.nn.relu

X = tf.placeholder(tf.float32, shape=[None, n_in])

initializer = tf.variance_scaling_initializer()

w1 = tf.Variable(initializer([n_in, n_hid1]), dtype=tf.float32)
w2 = tf.Variable(initializer([n_hid1, n_hid2]), dtype=tf.float32)
w3 = tf.Variable(initializer([n_hid2, n_hid3]), dtype=tf.float32)
w4 = tf.Variable(initializer([n_hid3, n_out]), dtype=tf.float32)

b1 = tf.Variable(tf.zeros(n_hid1))
b2 = tf.Variable(tf.zeros(n_hid2))
b3 = tf.Variable(tf.zeros(n_hid3))
b4 = tf.Variable(tf.zeros(n_out))

hid_l1 = actf(tf.matmul(X, w1)+b1)
hid_l2 = actf(tf.matmul(hid_l1, w2)+b2)
hid_l3 = actf(tf.matmul(hid_l2, w3)+b3)
out_l = actf(tf.matmul(hid_l3, w4)+b4)

loss = tf.reduce_mean(tf.square(out_l - X))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epoch = 10
b_size = 150

with tf.Session() as sess:
    sess.run(init)

    for e in range(n_epoch):
        n_batches = mnist.train.num_examples // b_size

        for _ in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(b_size)
            sess.run(train, feed_dict={X:X_batch})

        training_loss = loss.eval(feed_dict={X:X_batch})

        print("EPOCH: {} LOSS: {}".format(e, training_loss))

    saver.save(sess, "./stacked_autoencoder")

n_img = 10

import matplotlib.pyplot as plt

with tf.Session() as sess:
    saver.restore(sess, "./stacked_autoencoder")

    results = out_l.eval(feed_dict={X:mnist.test.images[:n_img]})

    f,a = plt.subplots(2,10, figsize=(20,4))
    for i in range(n_img):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(results[i], (28, 28)))

    plt.savefig('stacked_autoencoder.png')

