import sys
from matplotlib import pyplot as plt
from keras.models import model_from_json


def plot_weights(model):
    conv_weights = model.get_weights()[0]
    plot_width = 8
    plot_height = int(len(conv_weights)/plot_width)
    for i in range(plot_width*plot_height):
        plt.subplot(plot_height, plot_width, i+1)
        plt.imshow(conv_weights[i,0],cmap='hot',interpolation='none')
        plt.axis('off')
    plt.show()
    return


if __name__ == '__main__':
    json_file = open(sys.argv[1], 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(sys.argv[2])
    plot_weights(model)
    pass

