from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.
x_test = x_test / 255.
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
input_shape = x_train[0, :, :, :].shape

model_input = Input(shape=input_shape)

# mlpconv block1
x = Conv2D(32, (5, 5), activation='relu', padding='valid')(model_input)
x = Conv2D(32, (1, 1), activation='relu')(x)
x = Conv2D(32, (1, 1), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.5)(x)

# mlpconv block2
x = Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
x = Conv2D(64, (1, 1), activation='relu')(x)
x = Conv2D(64, (1, 1), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.5)(x)

# mlpconv block3
x = Conv2D(128, (3, 3), activation='relu', padding='valid')(x)
x = Conv2D(32, (1, 1), activation='relu')(x)
x = Conv2D(10, (1, 1), activation='relu')(x)

x = GlobalAveragePooling2D()(x)
x = Activation(activation='softmax')(x)

model = Model(model_input, x, name='nin_cnn')

model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])

_ = model.fit(x=x_train, y=y_train, batch_size=32,
                            epochs=200, verbose=1, validation_split=0.2)
