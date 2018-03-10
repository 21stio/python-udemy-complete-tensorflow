import tensorflow as tf
import gym
import numpy as np


n_in = 4
n_hid = 4
n_out = 1

learning_rate = 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_in])

hid_l = tf.layers.dense(X, n_hid, activation=tf.nn.relu, kernel_initializer=initializer)

logits = tf.layers.dense(hid_l, n_out)
outputs = tf.nn.sigmoid(logits)

probs = tf.concat(axis=1, values=[outputs, 1 - outputs])

action = tf.multinomial(probs, num_samples=1)

y = 1 - tf.to_float(action)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)

gradients_and_variables = optimizer.compute_gradients(cross_entropy)

gradients = []
gradient_placeholders = []
grads_and_vars_feed = []

for g, v in gradients_and_variables:
    gradients.append(g)
    gp = tf.placeholder(tf.float32, shape=g.get_shape())
    gradient_placeholders.append(gp)
    grads_and_vars_feed.append((gp, v))

training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def helper_discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0

    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards

    return discounted_rewards


def discount_and_normalized_rewards(all_rewards, discount_rate):
    all_discounted_rewards = []

    for rewards in all_rewards:
        all_discounted_rewards.append(helper_discount_rewards(rewards, discount_rate))

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()

    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]


env = gym.make("CartPole-v0")

n_game_rounds = 10
max_game_steps = 1000
n_iters = 760
discount_rate = 0.9

with tf.Session() as sess:
    sess.run(init)

    for i in range(n_iters):
        print("On iteration: {}".format(i))

        all_rewards = []
        all_gradients = []

        for game in range(n_game_rounds):
            current_rewards = []
            current_gradients = []

            o = env.reset()

            for step in range(max_game_steps):
                action_v, gradients_v = sess.run([action, gradients], feed_dict={X: o.reshape(1, n_in)})

                o, reward, done, info = env.step(action_v[0][0])

                current_rewards.append(reward)
                current_gradients.append(gradients_v)

                if done:
                    break

            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalized_rewards(all_rewards, discount_rate)

        feed_dict = {}

        for gp_i, gp in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_i][step][gp_i]
                                      for game_i, rewards in enumerate(all_rewards)
                                      for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gp] = mean_gradients

        sess.run(training_op, feed_dict=feed_dict)

        print("SAVING GRAPH AND SESSION")

        meta_graph_def = tf.train.export_meta_graph(filename='models/my-policy-model.meta')
        saver.save(sess, 'models/my-policy-model')

observations = env.reset()
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('models/my-policy-model.meta')
    new_saver.restore(sess, 'models/my-policy-model')

    for x in range(500):
        env.render()

        action_v, gradients_v = sess.run([action, gradients], feed_dict={X: observations.reshape(1, n_in)})
        o, reward, done, info = env.step(action_v[0][0])