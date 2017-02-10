import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivate(x):
    return sigmoid * (1 - sigmoid)

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5,
                                       (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate

        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = sigmoid

    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #### Implement the backward pass here ####
        ### Backward pass ###

        output_error = targets_list - final_outputs

        # TODO: Output error
        output_grad = error * output_error * (1 - output_error)  # error gradient in output

        # TODO: Backpropagated error
        hidden_errors = np.dot(output_errors, self.weights_hidden_to_output) * hidden_outputs * (1 - hidden_outputs)  # errors propagated to the hidden layer
        hidden_grad = hidden_errors * hidden_inputs
        #
        # # TODO: Update the weights
        self.weights_hidden_to_output += self.learning_rate * output_grad * hidden_outputs # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.learning_rate * hidden_grad * inputs  # update input-to-hidden weights with gradient descent step


    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# Do some experiment here
# def test():
#     x = [2, 2]
#     w = [1, 3]
#     print(np.dot(w, x))

def train():
    ### Set the hyperparameters here ###
    epochs = 100
    learning_rate = 0.1
    hidden_nodes = 2
    output_nodes = 1

    N_i = train_features.shape[1]
    network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

    losses = {'train':[], 'validation':[]}
    for e in range(epochs):
        # Go through a random batch of 128 records from the training data set
        batch = np.random.choice(train_features.index, size=128)
        for record, target in zip(train_features.ix[batch].values,
                                  train_targets.ix[batch]['cnt']):
            network.train(record, target)

        # Printing out the training progress
        train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
        val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
        sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
                         + "% ... Training loss: " + str(train_loss)[:5] \
                         + " ... Validation loss: " + str(val_loss)[:5])

        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)

        plt.plot(losses['train'], label='Training loss')
        plt.plot(losses['validation'], label='Validation loss')
        plt.legend()
        plt.ylim(ymax=0.5)

def predict():
    fig, ax = plt.subplots(figsize=(8,4))

    mean, std = scaled_features['cnt']
    predictions = network.run(test_features)*std + mean
    ax.plot(predictions[0], label='Prediction')
    ax.plot((test_targets['cnt']*std + mean).values, label='Data')
    ax.set_xlim(right=len(predictions))
    ax.legend()

    dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    _ = ax.set_xticklabels(dates[12::24], rotation=45)


def main():
    data_path = 'Bike-Sharing-Dataset/hour.csv'

    rides = pd.read_csv(data_path)
    print(rides.head())

    print(rides.shape)

    rides[:24*10].plot(x='dteday', y='cnt')
    # plt.show()

    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for each in dummy_fields:
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)

    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                      'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    data = rides.drop(fields_to_drop, axis=1)
    print(data.head())

    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}
    for each in quant_features:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean)/std

    # Save the last 21 days
    test_data = data[-21*24:]
    data = data[:-21*24]

    # Separate the data into features and targets
    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

    # Hold out the last 60 days of the remaining data as a validation set
    train_features, train_targets = features[:-60*24], targets[:-60*24]
    val_features, val_targets = features[-60*24:], targets[-60*24:]

    print('training set: %d' % train_features.shape[0])
    print('validation set: %d' % val_features.shape[0])
    print('test set %d' % test_features.shape[0])
    print(type(train_features.values))
    for row in train_features.values:
        # print(row[None,:]) # Each row is just a list of type ndarray, so we need to convert it to matrix type
        break


#test()
# main()

# x = [1, 3, 3, 4]
# y = [1, 2, 3]
# # x = [1, 2, 3, 4]
# xd = np.matrix(x)
# yd = np.matrix(y)
#
# print(xd.T * yd)

# res = xd * yd[:,None]
# print(res)

# d = d.T
# d = d[:, None]
# print(d.shape)
# print(d[:, None])
# print(d.values)