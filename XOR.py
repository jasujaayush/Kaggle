import tensorflow as tf
import matplotlib.pyplot as plt

data = []
for n_hidden in range(2, 7, 2):
	bias_size = 0
	input_size = 2 + bias_size
	x_ = tf.placeholder(tf.float32, shape=[4,input_size], name="x-input")        
	y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input")
	Theta1 = tf.Variable(tf.random_uniform([input_size,n_hidden], -1, 1), name="Theta1")
	Theta2 = tf.Variable(tf.random_uniform([n_hidden,1], -1, 1), name="Theta2")
	A2 = tf.sigmoid(tf.matmul(x_, Theta1))                              
	Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2))    
	cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) +  ((1 - y_) * tf.log(1.0 - Hypothesis)))*-1) 
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost) 
	#train_step = tf.train.AdamOptimizer(0.01).minimize(cost)
	XOR_X = [[0,0],[0,1],[1,0],[1,1]]                           
	XOR_Y = [[0],[1],[1],[0]]

	init = tf.initialize_all_variables()                                
	sess = tf.Session()                                                 
	sess.run(init) 

	c = []
	for i in range(1000000):                                             
		sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})      
		if i % 10000 == 0:
			e.append(i)
			sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y})
			t = sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y})
			c.append(t)
			#print('Epoch ', i)
			#print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
			#print('Theta1 ', sess.run(Theta1))
			#print('Theta2 ', sess.run(Theta2))
			#print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
	c.append(sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
	print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
	data.append(c)	
i = 2
for d in data:
	plt.plot(d, label = i)
	i += 2
plt.legend(loc = 'upper right')
plt.xlabel('Iterations/1000')
plt.ylabel('Cost function')
plt.title('Cost vs Iterations for multiple hidden nodes')