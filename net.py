import numpy as np

# extracted from youtube - 4 min neural net example by SRAVAL
# simple neural net
# each row is a different training example, each column represents a different neuron

# sigmoid - generates value between 0 and 1
def nonlin(x, deriv=False):
    if deriv == True:
        return x * (1-x)

    return 1/(1 + np.exp(-x))


# 4 training examples with 3 input neurons each
# input data
X = np.array([
        [0,0,1],
        [0,1,1],
        [1,0,1],
        [1,1,1]
    ])


# one output neuron each (one for each example)
# output data
Y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

# seed random number to make them deterministic
# means that give random numbers that are generated the same starting point or seed so that we'll get the same sequence of generated numbers every time we run our program
np.random.seed(1)

# create synapses matrices
# synapses are connections between each neuron in one layer to every neuron in the next layer
# we're doing a 3 layered network, we need 2 synapse matrices
# each synapse has a random weight assigned to it
syn0 = 2 * np.random.random((3,4)) - 1
syn1 = 2 * np.random.random((4,1)) - 1

# begin the training code
# for loop to iterate over the training code to optimize the network for the given data set
for j in range(60000):
    # start by creating our first layer - it's just our input data
    l0 = X
    # now comes the prediction step => perform matrix multiplication (dot product between each layer and its synapse)
    # then we'll run our sigmoid function (activation function) on all the values in the matrix to create the next layer
    l1 = nonlin(np.dot(l0, syn0))
    # the next layer contains the prediction of the output data => then we do the same thing (dot product) to get next layer which is a more refined prediction
    l2 = nonlin(np.dot(l1, syn1))

    # now tha we have prediction of the ouput value in layer 2, let's compare to the expected output data => substraction to get error rate
    l2_error = Y - l2

    # print average error rate at a set interval to make sure it goes time every time
    if (j % 10000) == 0:
        print('Error', str(np.mean(np.abs(l2_error))))

    # next we'll multiply our error rate by result of the sigmoid function
    # the sigmoid function is used to get the derivative of our output prediction from layer2
    # this will give us a delta => this we'll use to reduce the error rate of our predictions when we update our synapses every iteration
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # then we want to see how much layer1 contributed to the error in layer2
    # This is called back propagation. we'll get this by multiplying layer2's delta by synapse one's transpose
    l1_error = l2_delta.dot(syn1.T)

    # then we'll get layer1's delta by multiplying it's error by the result of our sigmoid function
    l1_delta = l1_error * nonlin(l1, deriv=True)

    # now that we have deltas for each of our layers, we can use them to update our synapse weights to reduce the error rate more and more every iteration
    # this is an algorithm called gradient descent
    # update synapse weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

# if we run this, we can see that the error rate decreases every iteration and the predicted output is very close to the actual output
print("output after training")
print(l2)

