import itertools
import numpy as np
import pandas as pd
import glob
import os
import time
from matplotlib import pyplot as plt
from keras.utils import np_utils

import train_mnist


def train_model(model_type, train_x, train_y, test_x=None, num_epoch=5):
    if model_type == 'svm':
        from sklearn import svm
        train_x = train_x.reshape(train_x.shape[0], -1)
        model = svm.SVC()
        time1 = time.perf_counter()
        model.fit(train_x, train_y)
        time2 = time.perf_counter()
    elif model_type == 'cnn':
        nb_classes = 10
        train_y = np_utils.to_categorical(train_y, nb_classes)
        model = train_mnist.create_model(nb_classes)
        time1 = time.perf_counter()
        model.fit(train_x, train_y, nb_epoch=num_epoch, verbose=2)
        time2 = time.perf_counter()
    else:
        return None, None
    print("\nTraining time: %f\n" % (time2 - time1))
    prediction = None
    if test_x is not None:
        prediction = model.predict(test_x)
        pass
    return model, prediction


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_plot=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_plot:
        path = os.path.join(".", "%s.png" % (title.replace(' ', '_')))
        plt.savefig(path)
        pass
    # plt.show()
    return


def extract_data(data_location, skip_fleisch=False):
    files = glob.glob(os.path.join(data_location, '2016*.csv'))

    skin_samples = []
    not_skin_samples = []
    for file in files:
        data_raw = pd.read_csv(file, delimiter=';', na_values=0).as_matrix()[:-10]
        dimension = data_raw.shape[0]
        is_skin = 0
        if skip_fleisch and "Fleisch" in file:
            continue
        if "Referenz-Haut" in file or 'skin' in file:
            is_skin = 1
            pass
        for material_class_index in range(1, data_raw.shape[1]):
            material_data = np.append(data_raw[:, material_class_index].reshape(1, dimension), is_skin)
            if is_skin == 1:
                skin_samples.append(material_data)
            else:
                not_skin_samples.append(material_data)
                pass
            pass
        pass

    skin_samples = np.asarray(skin_samples)
    not_skin_samples = np.asarray(not_skin_samples)
    return skin_samples, not_skin_samples