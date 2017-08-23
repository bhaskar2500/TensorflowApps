'''
MNIST dataset
- 60,000 training sample
- 10,000 training samples
- each image is 28x28 pixels
'''


'''
input data (*) weight -> Hidden Layer 1 (activation function) (*) weights -> Hidden Layer 2 (activation function) (*) weights -> Output Layer

Feed Forward neural Network

Compare output to intended out put using a cost function ( Cross entropy)
OPtimization function (Adam optimizers,stochaistic gradient desct, adagrad)

backpropagation
feedforward + backpropagation = epoch - 1 cycle (n times till the cosst is lowered close to 0)

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#for multi class clasification
mnist = input_data.read_data_sets('/tmp/data',one_hot = True)
#currently 10 classes 0-9
# 0 = [1,0,0,0,0,0,0,0,0,0,]

#Nodes in each layer 
total_nodes_hl1 = 500
total_nodes_hl2 = 500
total_nodes_hl3 = 500

total_classes = 10
#100 images at a time
batch_size = 100

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

#neural net
def neural_network_model(data):
	#(input data * weight) + biases
	hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784,total_nodes_hl1])), 
					  'biases':tf.Variable(tf.random_normal([total_nodes_hl1]))}
	hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([total_nodes_hl1,total_nodes_hl2])), 
					  'biases':tf.Variable(tf.random_normal([total_nodes_hl2]))}
	hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([total_nodes_hl2,total_nodes_hl3])), 
					  'biases':tf.Variable(tf.random_normal([total_nodes_hl3]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([total_nodes_hl3,total_classes])), 
					  'biases':tf.Variable(tf.random_normal([total_classes]))}

	layer_1 = tf.add(tf.matmul(data,hidden_layer_1['weights']), hidden_layer_1['biases'])
	#rectified linear as threshold function
	layer_1 = tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1,hidden_layer_2['weights']), hidden_layer_2['biases'])
	layer_2 = tf.nn.relu(layer_2)

	layer_3 = tf.add(tf.matmul(layer_2,hidden_layer_3['weights']), hidden_layer_3['biases'])
	layer_3 = tf.nn.relu(layer_3)

	output = tf.matmul(layer_3,output_layer['weights']) + output_layer['biases']

	return output


#training neural net
def train_neural_net(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	#default learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)	

	#cycles of feed forward + back prop
	hm_epochs = 15
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size)
				
				_, c = sess.run([optimizer, cost], feed_dict= {x:epoch_x, y:epoch_y})
			
				epoch_loss += c	
			print('Epoch', epoch, 'completed out of', hm_epochs,'loss:',epoch_loss,'batch_size:', batch_size)

		correcct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		print(correcct)
		accuracy = tf.reduce_mean(tf.cast(correcct,'float'))
		print(accuracy)
		print('Accuracy:', accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
	#	print(sess.run(y, feed_dict={x: mnist.test.images}))

train_neural_net(x)