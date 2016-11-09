import numpy as np
import math
import random
import csv
import argparse
import simple_nn as nn
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='A simple fully-connected neural net backprop implementation')
parser.add_argument('-data', '--data', default="data/data_3class.csv", help='directory to data', type=str)
parser.add_argument('-n', '--classes', default=3, help='number of output classes', type=int)
parser.add_argument('-l', '--layers', default=0, help='number of layers', type=int)
parser.add_argument('-d', '--dimensions', nargs = '*', default=[], help='number of nodes in each layer', type=int)

LEARNING_RATE = 0.001

def shuffle_in_unison(a, b):
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def softmax(x, deriv=False):
	offset = np.amax(x)
	x = x - offset + 1e-16
	e_vals = np.exp(x)

	return np.true_divide(e_vals, np.sum(e_vals))

def cross_entropy(pred, actual):
	pred = pred + 1e-16
	log_pred = np.log(pred)
	dot_prod = np.dot(log_pred, actual)

	return -1.0*dot_prod

def parse_csv(filename):
	# Split into X and Y values
	X_vals = []
	Y_vals = []
	with open(filename, 'rb') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			split_row = row[0].split(" ")
			split_row = [float(num) for num in split_row]
			X_vals.append(split_row[:-1])
			Y_vals.append(split_row[-1])
	return (X_vals, Y_vals)

def build_nn(inp_size, num_classes, inp_layers, inp_dims):
	if len(inp_dims) != inp_layers:
		raise ValueError("Number of layers %s does not match the length of the dimension list %s" % (inp_layers, len(inp_dims)))

	network = {}

	# Create an input layer
	input_layer = nn.Layer(inp_size)
	network[0] = input_layer

	# Otherwise, create a weight matrix for each layer
	for i in xrange(inp_layers):
		prev_layer = network[i]
		fc_layer = nn.FCLayer(inp_dims[i], prev_layer)
		network[i+1] = fc_layer

	# Create an output layer
	output_layer = nn.FCLayer(num_classes, network[inp_layers], activation_fn = softmax)
	network[inp_layers + 1] = output_layer

	return network

def backprop(network, train_set, val_set, test_set, max_iters = 10000):
	num_layers = len(network.keys())
	num_classes = network[num_layers - 1].get_dim()
	best_ce = float('inf')
	best_network = {}
	best_acc = 0
	prev_ce = 0
	for i in xrange(max_iters):
		# Randomly select a data point
		#rand_index = random.randint(0, len(data[0])-1)
		#rand_index = 0
		# train_set = shuffle_in_unison(train_set[0], train_set[1])
		for j in xrange(len(train_set[0])):
			rand_index = random.randint(0, len(train_set[0])-1)
			rand_x = train_set[0][rand_index]
			rand_y = train_set[1][rand_index]

			# (optimal y prediction is 1.0 @ y, 0 elsewhere)
			opt_y = np.zeros((num_classes,))
			opt_y[int(rand_y)] = 1

			network[0].set_values(rand_x)
			cur_value = network[0].get_values()
			for ind in xrange(1, num_layers):
				network[ind].forward_prop()
				cur_value = network[ind].get_values()

			output_value = network[num_layers - 1].get_values()

			# How much were we off?
			y_deriv = output_value - opt_y #cross-entropy derivative

			# backprop each of the weights
			for ind in xrange(num_layers - 1, 0, -1):
				cur_layer = network[ind]
				cur_weights = cur_layer.get_weights()
				inp_values = cur_layer.get_input().get_values()
				prev_layer = cur_layer.get_input()
				derivatives = prev_layer.get_activation()(prev_layer.get_input_values(), deriv=True)

				cur_layer.set_biases(cur_layer.get_biases() - LEARNING_RATE*y_deriv)

				cur_layer.set_weights(cur_weights - LEARNING_RATE*np.dot(np.atleast_2d(inp_values).T, np.atleast_2d(y_deriv)))

				y_deriv = np.multiply(derivatives, np.dot(cur_weights, y_deriv))

		
		total_ce, acc = eval_network(network, val_set)
		if total_ce < best_ce:
			print "Updating best loss: ", total_ce
			print "Validation acc: ", acc
			best_ce = total_ce
			for key in network.keys():
				best_network[key] = network[key]

		if abs(prev_ce - total_ce) < 1e-10:
			break

		prev_ce = total_ce

		if i % 1000 == 0:
			print "Cross-entropy error " + str(i), best_ce

	print "Beginning evaluation..."
	plot_network(best_network, test_set)
	return best_network

def eval_network(net, data):
	X_data = data[0]
	Y_data = data[1]
	num_layers = len(network.keys())
	num_classes = net[num_layers - 1].get_dim()

	accuracy = 0
	total_ce = 0

	for i in xrange(len(X_data)):
		cur_x = X_data[i]
		cur_y = Y_data[i]

		# (optimal y prediction is 1.0 @ y, 0 elsewhere)
		opt_y = np.zeros((num_classes,))
		opt_y[int(cur_y)] = 1

		network[0].set_values(cur_x)
		cur_value = network[0].get_values()
		for ind in xrange(1, num_layers):
			network[ind].forward_prop()
			cur_value = network[ind].get_values()

		output_value = network[num_layers - 1].get_values()
		if abs(sum(output_value) - 1) > 1e-7:
			print sum(output_value)

		y_error = cross_entropy(output_value, opt_y) #cross-entropy error
		total_ce += y_error
		classification = np.argmax(output_value)
		if classification == int(cur_y):
			accuracy += 1
	acc = accuracy/float(len(X_data))

	return total_ce/float(len(X_data)), acc

def plot_network(net, data):
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	X_data = data[0]
	Y_data = data[1]
	num_layers = len(network.keys())
	num_classes = net[num_layers - 1].get_dim()

	classifications = {}
	for j in xrange(num_classes):
		classifications[j] = []
	accuracy = 0

	for i in xrange(len(X_data)):
		cur_x = X_data[i]
		cur_y = Y_data[i]

		network[0].set_values(cur_x)
		cur_value = network[0].get_values()
		for ind in xrange(1, num_layers):
			network[ind].forward_prop()
			cur_value = network[ind].get_values()

		output_value = network[num_layers - 1].get_values()
		classification = np.argmax(output_value)
		if classification == int(cur_y):
			accuracy += 1

		classifications[classification].append(cur_x)

	print "Accuracy: ", accuracy/float(len(X_data))

	for k in xrange(num_classes):
		xs = [thing[0] for thing in classifications[k]]
		ys = [thing[1] for thing in classifications[k]]

		color = colors[k]

		plt.plot(xs, ys, color + 'o')

	plt.show()

def load_from_txt(name):
	# load data from csv files
	train = np.loadtxt('data/data'+name+'_train.csv', dtype=np.float64)
	trainX = np.array(train[:,0:2])
	trainY = train[:,2:3]
	trainY[trainY < 0] = 0
	train_data = (trainX, np.squeeze(trainY))

	validate = np.loadtxt('data/data'+name+'_validate.csv', dtype=np.float64)
	valX = validate[:,0:2]
	valY = validate[:,2:3]
	valY[valY < 0] = 0
	val_data = (valX, np.squeeze(valY))

	test = np.loadtxt('data/data'+name+'_test.csv', dtype=np.float64)
	testX = test[:,0:2]
	testY = test[:,2:3]
	testY[testY < 0] = 0
	test_data = (testX, np.squeeze(testY))

	return train_data, val_data, test_data

def load_mnist():
	trainXs = []
	trainYs = []
	valXs = []
	valYs = []
	testXs = []
	testYs = []
	for i in xrange(10):
		name = str(i)

		train = np.genfromtxt('data/mnist_digit_'+name+'.csv', dtype=np.float64, max_rows=200)
		train = 2*np.true_divide(train, 255) - 1
		trainXs.append(train)
		trainYs.append(i*np.ones((200,)))

		val = np.genfromtxt('data/mnist_digit_'+name+'.csv', dtype=np.float64, skip_header=200, max_rows=150)
		val = 2*np.true_divide(val, 255) - 1
		valXs.append(val)
		valYs.append(i*np.ones((150,)))
		
		test = np.genfromtxt('data/mnist_digit_'+name+'.csv', dtype=np.float64, skip_header=350, max_rows=150)
		test = 2*np.true_divide(test, 255) - 1
		testXs.append(test)
		testYs.append(i*np.ones((150,)))

	return (np.concatenate(trainXs), np.concatenate(trainYs)), (np.concatenate(valXs), np.concatenate(valYs)), (np.concatenate(testXs), np.concatenate(testYs))

if __name__ == '__main__':
	args = parser.parse_args()
	data = parse_csv(args.data)
	# parameters
	#train_data, val_data, test_data = load_from_txt('1')
	#train_data, val_data, test_data = load_mnist()

	num_layers = args.layers
	dim_list = args.dimensions
	num_classes = args.classes

	X_dim = len(data[0][0])

	network = build_nn(X_dim, num_classes, num_layers, dim_list)
	print "Beginning training..."
	network = backprop(network, data, data, data, max_iters = 1000)