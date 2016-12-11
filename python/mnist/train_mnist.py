#!/usr/bin/env python3
import sys
import time
import numpy as np
#from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split


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

    print(acc/len(X_test))
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


def main(num_epoch):
    date_string = time.strftime("%Y%m%d")
    np.random.seed(1337)  # for reproducibility
    mnist = fetch_mldata('MNIST original')

    data = mnist.data.reshape((len(mnist.data),1,28,28))
    data = data.astype('float32')
    data = data/255

    testset_ratio = 0.33
    X_train, X_test, Y_train, Y_test = train_test_split(data, mnist.target.astype('int'), test_size=testset_ratio)

    # convert class vectors to binary class matrices
    from keras.utils import np_utils
    nb_classes = 10
    batch_size = 128
    nb_epoch = num_epoch
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

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test), callbacks=[csv_logger])
    model.save_weights(base_name + "_weights.h5")
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    acc, w, c_matrix = test_network(model, X_test, Y_test)
    print(c_matrix)
    return


if __name__ == '__main__':
    num_epoch = int(sys.argv[1])
    main(num_epoch)
