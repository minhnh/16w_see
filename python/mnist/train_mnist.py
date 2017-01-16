#!/usr/bin/env python3
import sys
import time
import numpy as np
#from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn import svm
import util


def categorial_to_number(label):
    shape = np.shape(label)
    v = np.arange(shape[1]).reshape(-1,1)
    return np.dot(label,v)

def test_network(model, X_test, Y_test):
    acc = 0
    wrong_list = list()
    predicts = np.round(model.predict(X_test))
    predicts = categorial_to_number(predicts)
    Y_test = categorial_to_number(Y_test)

    c_matrix = confusion_matrix(Y_test, predicts)
    for n in range(len(X_test)):
        if (np.allclose(predicts[n],Y_test[n])):
            acc = acc + 1.0
        else:
            #Wrong index, predicted label, true label
            wrong_list.append((n, predicts[n][0], Y_test[n][0]))
            pass
        pass

    acc /= len(X_test)
    print('Test accuracy: ', acc)
    return acc, wrong_list, c_matrix


def create_model(nb_classes, batch_size):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D

    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid', input_shape=(1,28,28)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def train_cnn(X_train, Y_train, X_test, Y_test, nb_epoch, date_string):
    # convert class vectors to binary class matrices
    from keras.utils import np_utils
    nb_classes = 10
    batch_size = 128
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    base_name = "mnist_%s" % date_string
    model = create_model(nb_classes, batch_size)
    model_json = model.to_json()
    model_file_name = base_name + "_model.json"
    with open(model_file_name, "w") as json_file:
        json_file.write(model_json)
        pass

    from keras.callbacks import CSVLogger
    csv_logger = CSVLogger("%s_training.log" % base_name, append=False)

    time1 = time.perf_counter()
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=2, validation_data=(X_test, Y_test), callbacks=[csv_logger])
    time2 = time.perf_counter()
    print("Train time: %f s" % (time2 - time1))
    model.save_weights(base_name + "_weights.h5")

    acc, w, c_matrix = test_network(model, X_test, Y_test)
    # print(c_matrix)
    util.plot_confusion_matrix(c_matrix, list(range(10)), save_plot=True, title="confusion matrix CNN")
    return


def train_svm(train_x, train_y, test_x, test_y):
    model = svm.SVC()
    time1 = time.perf_counter()
    model.fit(train_x, train_y)
    time2 = time.perf_counter()
    print("Train time: %f s" % (time2 - time1))
    prediction = model.predict(test_x)
    c_matrix = confusion_matrix(test_y, prediction, labels=list(range(10)))
    accuracy = np.sum(np.diag(c_matrix)) / np.sum(c_matrix)
    print("Accuracy: %f" % accuracy)
    util.plot_confusion_matrix(c_matrix, list(range(10)), save_plot=True, title="confusion matrix SVM")
    return


def main(num_epoch, model):
    date_string = time.strftime("%Y%m%d")
    np.random.seed(1337)  # for reproducibility
    mnist = fetch_mldata('MNIST original')

    data = mnist.data.astype('float32')
    data /= 255

    testset_ratio = 0.33
    X_train, X_test, Y_train, Y_test = train_test_split(data, mnist.target.astype('int'), test_size=testset_ratio)
    if model == 'cnn':
        X_train = X_train.reshape((len(X_train), 1, 28, 28))
        X_test = X_test.reshape((len(X_test), 1, 28, 28))
        train_cnn(X_train, Y_train, X_test, Y_test, num_epoch, date_string)
    elif model == 'svm':
        train_svm(X_train, Y_train, X_test, Y_test)
        pass

    return


if __name__ == '__main__':
    num_epoch, model = int(sys.argv[1]), sys.argv[2]
    main(num_epoch, model)
