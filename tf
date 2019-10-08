import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
    inital=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

sess=tf.InteractiveSession()

#图像处理

x=tf.placeholder(tf.float32,shape=[None,784])
y=tf.placeholder(tf.float32,shape=[None,10])

x_image=tf.reshape(x,[-1,28,28,1])

#第一层CNNC层
#number of nodes
w_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])

h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1=max_pool(h_conv1)

#第二层CNN层
w_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool(h_conv2)

#全连接层
w_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

#dropout层
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#第二个全连接层
w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

#损失函数与训练算法
cross_entropy=-tf.reduce_sum(y*tf.log(y_conv))
#learining rate MomentumOptimizer
train=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction=tf.equal(tf.arg_max(y_conv,1),tf.arg_max(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess.run(tf.global_variables_initializer())

#iteration
for i in range(20000):
#batch
    batch=mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y: batch[1], keep_prob: 1.0})
        print ("step %d, training accuracy %g" % (i, train_accuracy))

    train.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

    print ("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
