from keras.layers import Activation, Input, Dense, Flatten, Concatenate, Conv2D
from keras.models import Model
from keras.utils import plot_model

from utils import OUTPUT_SHAPES


class AtariModel(Model):
    def __init__(self, activations):
        # Добавить flat things
        # Добавить 1x1 convs
        # Возможно, убрать по первых 2 активации

        screen_shape = (64, 64, 17)
        minimap_shape = (64, 64, 7)

        screen = Input(shape=screen_shape)
        conv_screen_1 = Conv2D(filters=17, kernel_size=(1, 1))(screen)
        conv_screen_1 = Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4))(conv_screen_1)
        if activations:
            conv_screen_1 = Activation('relu')(conv_screen_1)
        conv_screen_2 = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2))(conv_screen_1)
        if activations:
            conv_screen_2 = Activation('relu')(conv_screen_2)
        flat_screen = Flatten()(conv_screen_2)

        minimap = Input(shape=minimap_shape)
        conv_minimap_1 = Conv2D(filters=7, kernel_size=(1, 1))(minimap)
        conv_minimap_1 = Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4))(conv_minimap_1)
        if activations:
            conv_minimap_1 = Activation('relu')(conv_minimap_1)
        conv_minimap_2 = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2))(conv_minimap_1)
        if activations:
            conv_minimap_2 = Activation('relu')(conv_minimap_2)
        flat_minimap = Flatten()(conv_minimap_2)

        concat = Concatenate()([flat_screen, flat_minimap])

        outputs = []
        for output_info in OUTPUT_SHAPES:
            output_shape = output_info[1]
            output = Dense(output_shape)(concat)
            output = Activation('softmax')(output)
            outputs.append(output)

        super(AtariModel, self).__init__(inputs=(screen, minimap), outputs=outputs)

    def plot_model(self):
        plot_model(self, to_file='my_graph.png')

if __name__ == '__main__':
    model = AtariModel()
    model.plot_model()
