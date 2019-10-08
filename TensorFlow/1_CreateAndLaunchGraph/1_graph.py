import tensorflow as tf

# constant 1x2
m1 = tf.constant([[3, 3]])

# constant 2x1
m2 = tf.constant([[2], [3]])

# m1 x m2
product = tf.matmul(m1, m2)
print(product)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)

