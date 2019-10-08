import tensorflow as tf
import numpy as np

# use numpy generate 100 random points (SAMPLE)
x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

# build a linear model
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

# square cost function
loss = tf.reduce_mean(tf.square(y_data-y))

# define a gradient descending optimizer
optimizer = tf.train.GradientDescentOptimizer(0.2)

# minimize cost function
train = optimizer.minimize(loss) # Op

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(401):
        sess.run(train)
        if epoch%20 == 0:
            print(epoch, sess.run([k, b]))
