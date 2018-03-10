import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("../03-Convolutional-Neural-Networks/MNIST_data/", one_hot=True)

plt.imshow(mnist.train.images[13].reshape(28, 28))


def generator(z, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        hid_l1 = tf.layers.dense(inputs=z, units=128, activation=tf.nn.leaky_relu)

        hid_l2 = tf.layers.dense(inputs=hid_l1, units=128, activation=tf.nn.leaky_relu)

        out_l = tf.layers.dense(hid_l2, units=784, activation=tf.nn.tanh)

        return out_l


def discriminator(X, reuse=None):
    with tf.variable_scope('dis', reuse=reuse):
        hid_l1 = tf.layers.dense(inputs=X, units=128, activation=tf.nn.leaky_relu)

        hid_l2 = tf.layers.dense(inputs=hid_l1, units=128, activation=tf.nn.leaky_relu)

        out_l = tf.layers.dense(hid_l2, units=1, activation=tf.nn.sigmoid)

        return out_l


real_images = tf.placeholder(tf.float32, shape=[None, 784])
z = tf.placeholder(tf.float32, shape=[None, 100])

G = generator(z)
D_output_real, D_logits_real = discriminator(real_images)
D_output_fake, D_logits_fake = discriminator(G, reuse=True)


def loss_func(logits_in, labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))


D_real_loss = loss_func(D_logits_real, tf.ones_like(D_logits_real))
D_fake_loss = loss_func(D_logits_fake, tf.ones_like(D_logits_real))

D_loss = D_real_loss + D_fake_loss
G_loss = loss_func(D_fake_loss, tf.ones_like(D_logits_fake))

learning_rate = 0.001

tvars = tf.trainable_variables()

d_vars = [v for v in tvars if 'dis' in v.name]
g_vars = [v for v in tvars if 'geb' in v.name]

D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=g_vars)

batch_size = 100
epochs = 1

init = tf.global_variables_initializer()

samples = []

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        n_batches = mnist.train.num_examples // batch_size

        for i in range(n_batches):
            batch = mnist.train.next_batch(batch_size)

            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images * 2 - 1

            batch_z = np.random.uniform(-1,1,size=(batch_size, 100))

            _ = sess.run(D_trainer, feed_dict={real_images: batch_images, z:batch_z})
            _ = sess.run(G_trainer, feed_dict={z: batch_z})

        print("ON EPOCH {}".format(epoch))

        sample_z = np.random.uniform(-1,1,size=(1,100))
        gen_sample = sess.run(generator(z,reuse=True), feed_dict={z:sample_z})

        samples.append(gen_sample)


import matplotlib.pyplot as plt

n = len(samples)
f,a = plt.subplots(1,n, figsize=(20,4))
for i in range(n):
    a[0][i].imshow(samples[i])