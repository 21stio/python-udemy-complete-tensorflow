import tensorflow as tf
from beeprint import pp
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def init_weights(shape, name="weights"):
    with tf.name_scope(name):
        init_random_dist = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_random_dist, name="weights")


def init_bias(shape, name="bias"):
    with tf.name_scope(name):
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals, name="bias")


def conv2d(x, W, name="conv2d"):
    with tf.name_scope(name):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x, name="pool"):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolution_layer(input_x, shape, name="convolution_layer"):
    with tf.name_scope(name):
        W = init_weights(shape)
        variable_summaries(W)
        b = init_bias([shape[3]])
        variable_summaries(b)
        return tf.nn.relu(conv2d(input_x, W) + b)


def normal_full_layer(input_layer, size, name="full_layer"):
    with tf.name_scope(name):
        input_size = int(input_layer.get_shape()[1])
        W = init_weights([input_size, size])
        variable_summaries(W)
        b = init_bias([size])
        variable_summaries(b)
        return tf.matmul(input_layer, W) + b


with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=(None, 784), name="x")
    y_true = tf.placeholder(tf.float32, shape=(None, 10), name="y_true")
    hold_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('hold_prob', hold_prob)

x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope("network"):
    convo_1 = convolution_layer(x_image, shape=[5, 5, 1, 32])
    convo_1_pooling = max_pool_2by2(convo_1)

    convo_2 = convolution_layer(convo_1_pooling, shape=[5, 5, 32, 64])
    convo_2_pooling = max_pool_2by2(convo_2)

    convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7 * 7 * 64])
    full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

    full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

    y_pred = normal_full_layer(full_one_dropout, 10)

with tf.name_scope("train"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    tf.summary.scalar('cross_entropy', cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

steps = 100

summaries_dir = "/tmp/cnn_mnist/summary"

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(summaries_dir + '/train')
test_writer = tf.summary.FileWriter(summaries_dir + '/test')

with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(50)

        summary, _ = sess.run([merged, train], feed_dict={x: batch_x, y_true: batch_y, hold_prob: 1.0})
        train_writer.add_summary(summary, i)

        if i % 100 == 0:
            with tf.name_scope("eval"):
                print("ON STEP: {}".format(i))
                print("ACCURACY: ")

                matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

                accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
                tf.summary.scalar('accuracy', accuracy)

                summary, accuracy = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_true: mnist.test.labels, hold_prob: 1.0})
                test_writer.add_summary(summary, i)

                pp(accuracy)

    train_writer.add_graph(sess.graph)

train_writer.flush()
test_writer.flush()

