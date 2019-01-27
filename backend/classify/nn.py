from keras.layers import Input,Dense,Conv2D,Flatten,Dropout,MaxPooling2D
from keras.models import Model

def vgg_fm(input_shape):
    input_tensor=Input(shape=input_shape)
    x = Conv2D(32, (7, 7), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', padding='same')(x)

    # Block 2
    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', padding='same')(x)

    # Block 3
    x = Conv2D(128, (1, 1), activation='relu', padding='same', name='block3_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', name='predictions')(x)
    return Model(inputs=[input_tensor],outputs=[x])
