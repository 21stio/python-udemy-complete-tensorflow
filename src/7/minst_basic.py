import tensorflow as tf
from beeprint import pp

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MINST_data/", one_hot=True)

pp(mnist.train.images.shape)

img = mnist.train.images[0].reshape(28, 28)

x = tf.placeholder(tf.float32, shape=[None, 784], name="x")

W = tf.Variable(tf.zeros([784, 10]), name="weights")
b = tf.Variable(tf.zeros([10]), name="bias")

y = tf.matmul(x, W) + b

y_true = tf.placeholder(tf.float32, [None, 10], name="y_true")

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y), name="cross_entropy")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)

        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})


    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))

    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    print(sess.run(acc, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))

    writer = tf.summary.FileWriter("/tmp/minst_basic")
    writer.add_graph(sess.graph)
    writer.flush()

