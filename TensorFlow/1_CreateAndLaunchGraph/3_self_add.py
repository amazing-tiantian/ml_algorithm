import tensorflow as tf

state = tf.Variable(0, name='counter')
self_add = tf.add(state, 1)
update = tf.assign(state, self_add)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))