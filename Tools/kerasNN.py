from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.optimizers as op
from keras.initializers import random_normal
from sklearn.metrics import roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt
import itertools
import numpy as np
from os.path import join


class KerasNeuralNetwork(object):

    def __init__(self, input_neuron_num=1, output_neuron_num=1, hidden_neuron_nums=[1],
                 hidden_activation='relu', output_activation='sigmoid',
                 dropout=0.2, loss_function='categorical_crossentropy', optimiser='adam',
                 learning_rate=0.01, decay_rate=1e-6, momentum=0.2, nesterov=True,
                 load_model=False, model_dir='/g/data/k10/dr1709/code/Models', model_name='NN.h5'):
        """
        Initialises a simple artificial neural network (ANN) object using the
        keras module with the specified number of input layer neurons, output
        layer neurons and hidden layer neurons. The ANN can also be customised
        with your choice of hidden layer activation function, output layer
        function, dropout rate, loss function, optimiser and learning rate. If
        using sgd for the optimiser, you can also customise the decay rate, momentum
        and whether to activate the Nesterov momentum.

        :param input_neuron_num: int type. Number of neurons in the input layer.
        :param output_neuron_num: int type. Number of neurons in the output layer.
        :param hidden_neuron_nums: list or numpy.ndarray of int types. A list or array
                                   of integers, each describing the number of neurons
                                   for the according hidden layer, e.g. [10, 20] would
                                   add two hidden layers, the first with 10 neurons and
                                   the second with 20 neurons.
        :param hidden_activation: str type. The activation function applied to the
                                  hidden layers of the ANN; defaults to 'relu'.
        :param output_activation: str type. The activation function applied to the
                                  output layer of the ANN; defaults to 'sigmoid'.
        :param dropout: float type. The fraction of neurons from the previous layer
                        to be dropped. This helps to prevent overfitting; defaults
                        to 0.2.
        :param loss_func: str type. A loss function from keras database; defaults to
                          'categorical_crossentropy', as model is designed for a
                          multi-class ANN using the sigmoid function.
        :param optimiser: str type. keras.optimizer type. Optimiser used to train the
                          ANN; defaults to Adam optimiser.
        :param learning_rate: float type. Defines the step taken by the gradient descent
                              in response to fitting; defaults to 0.01.
        :param decay_rate: float type. Defines how quickly the gradient descent steps
                           decrease in magnitude whilst fitting.
        :param momentum: float type. 'Parameter that accelerates SGD in the relevant
                         direction and dampens oscillations.'
                         - from https://keras.io/optimizers/#sgd
        :param nesterov: boolean type. 'Whether to apply Nesterov momentum.'
                         - from https://keras.io/optimizers/#sgd
        """
        if load_model:
            if model_name[-3:] != '.h5':
                model_name += '.h5'
            from keras.models import load_model
            self.model = load_model(join(model_dir, model_name))
        else:
            if type(hidden_neuron_nums) != type([]) and type(hidden_neuron_nums) != type(np.array([])):
                raise Exception('Hidden layer information must be given as a list\n'
                                + 'or numpy array of integers')
            if not isinstance(output_neuron_num, int) or not isinstance(input_neuron_num, int):
                raise Exception('Neuron numbers must be integers')
            if output_neuron_num <= 0 or input_neuron_num <=0:
                raise Exception('Neuron numbers must be greater than or equal to 1')
        ### Define class variables ###
            self.input_num = input_neuron_num
            self.output_num = output_neuron_num
            self.structure = [input_neuron_num] + list(hidden_neuron_nums) + [output_neuron_num]
            self.dropout = dropout
            self.hidden_activation = hidden_activation
            self.output_activation = output_activation
            self.loss_function = loss_function
            self.learning_rate = learning_rate
            self.decay = decay_rate
            if optimiser == 'adam':
                self.optimiser = op.Adam(lr=self.learning_rate)
            if optimiser == 'sgd':
                # self.decay = decay
                self.momentum = momentum
                self.nesterov = nesterov
                self.optimiser = op.SGD(lr=self.learning_rate,
                                        decay=self.decay,
                                        momentum=self.momentum,
                                        nesterov=self.nesterov)
        ### Define the model ###
            model = Sequential()
        ### Add input layer ###
            model.add(Dense(units=self.structure[1],
                            activation=self.hidden_activation,
                            input_shape=(self.structure[0],),
                            kernel_initializer=random_normal(stddev=0.1)))
            model.add(Dropout(self.dropout))
        ### Add hidden layers ###
            for l in range(2, len(self.structure)-1):
                model.add(Dense(units=self.structure[l],
                                activation=self.hidden_activation))
                # model.add(Dropout(self.dropout))
        ### Add output layer ###
            model.add(Dense(units=self.structure[-1],
                            activation=self.output_activation))
            self.model = model
            print(self.model.summary())

    def config(self):
        """
        Configure the ANN with the loss function and optimiser specified in the
        initialisation of the ANN object.

        :return: N/A
        """
        self.model.compile(loss=self.loss_function,
                           optimizer=self.optimiser,
                           metrics=['accuracy'])

    def save_model(self, base_dir, model_name):
        """
        Saves the ANN model as an HDF5 file.

        :param base_dir: str type. Path/to/directory where model is to be saved.
        :param model_name: str type. Name of the file without the file extension.
        :return: N/A
        """
        self.model.save(join(base_dir, model_name) + '.h5')
        print('Model saved')

    def train(self, train_inputs, train_classes, epochs, batch_size):
        """
        Trains the ANN.

        :param train_inputs: numpy.ndarray of input arrays.
                             NB// Values should all be normalised.
        :param train_classes: numpy.ndarray of output classes.
                              NB// Values should be 0 or 1.
        :param epochs: int type. Number of epochs to perform.
        :param batch_size: int type. The batch size for which to train the data.
        :return: N/A
        """
        ### Configure the ANN ###
        self.config()
        ### Train the ANN ###
        self.model.fit(train_inputs, train_classes, batch_size=batch_size,
                       epochs=epochs, verbose=1, shuffle=True)

    def evaluate(self, test_inputs, test_classes):
        """
        Evaluates the ANN.

        :param test_inputs: numpy.ndarray of input arrays.
                            NB// Values should all be normalised.
        :param test_classes: numpy.ndarray of output classes.
                             NB// Values should be 0 or 1.
        :return: Test score containing test loss and test accuracy.
        """
        score = self.model.evaluate(test_inputs, test_classes, verbose=1)
        return score

    def predict(self, input_arr):
        """
        Predicts the class of the input data from the input array.

        :param input_arr: numpy.ndarray of input arrays.
                          NB// Values should all be normalised.
        :return: numpy.ndarray of class predictions.
        """
        return self.model.predict(input_arr, verbose=1)

    def roc_curve(self, predictions, labels, show=False):
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        AUROC = auc(fpr, tpr)
        if show:
            plt.figure(2)
            plt.plot(fpr, tpr, 'r-')
            plt.xlabel('False positive rate')
            plt.ylabel('True negative rate')
            plt.title('ROC curve: variation of TPR with FPR')
            x = np.arange(0., 1., 0.001)
            y = x
            plt.plot(x, y, 'b--')
            plt.show()
        return fpr, tpr, thresholds, AUROC

    def confusion_matrix(self, features, labels, show=False):
        cm = confusion_matrix(labels, features)
        if show:
            plt.figure(1)
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, [0, 1])
            plt.yticks(tick_marks, [0, 1])
            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.show()
        return cm
