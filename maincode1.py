import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Flatten, Dense
from keras.models import Model, load_model
from keras.applications.mobilenet import MobileNet


ROOT_DIR = "D:\\3rd Year\\Sem 6\\Minor II\\data"
number_of_images = {}

for dir in os.listdir(ROOT_DIR):
    number_of_images[dir] = len(os.listdir(os.path.join(ROOT_DIR, dir)))
    print("", dir, "", number_of_images[dir])
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input


def preprocessingImage1(path):
    image_data = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, preprocessing_function=preprocess_input,
                                    horizontal_flip=True)
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='binary')
    return image


def preprocessionfImage2(path):
    image_data = ImageDataGenerator(preprocessing_function=preprocess_input)
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='binary')

    return image


path = "D:\\3rd Year\\Sem 6\\Minor II\\data\\train"
train_data = preprocessingImage1(path)

path = "D:\\3rd Year\\Sem 6\\Minor II\\data\\test"
test_data = preprocessionfImage2(path)
path = "D:\\3rd Year\\Sem 6\\Minor II\\data\\validation"
val_data = preprocessionfImage2(path)


base_model = MobileNet(input_shape=(224, 224, 3), include_top=False)
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping

for layer in base_model.layers:
    layer.trainable = False

# Define the top layers of the model
x = Flatten()(base_model.output)
x = Dense(units=1, activation='sigmoid')(x)

# Define the model
model = Model(base_model.input, x)

# Compile the model
model.compile(optimizer='rmsprop',
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])

# Define callbacks
mc = ModelCheckpoint(filepath="bestmodel.keras",
                     monitor='val_accuracy',
                     verbose=1,
                     save_best_only=True,
                     mode='max')
es = EarlyStopping(monitor="val_accuracy",
                   min_delta=0.01,
                   patience=5,
                   verbose=1,
                   mode='max')
callbacks = [mc, es]

# Training the model
hist = model.fit(train_data,
                 steps_per_epoch=10,
                 epochs=30,
                 validation_data=val_data,
                 validation_steps=16,
                 callbacks=callbacks)
from keras.preprocessing import image
import tensorflow as tf


def predictimage(path):
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    i = tf.keras.utils.img_to_array(img) / 255
    input_arr = np.array([i])
    input_arr.shape

    pred = model.predict(input_arr)
    if pred == 1:
        print("Not Affected")
    else:
        print("Affected")
    # display image
    plt.imshow(input_arr[0], vmin=0, vmax=255)
    plt.title("input Image")
    plt.show()


# It is the infected image sample
predictimage("D:\\3rd Year\\Sem 6\\Minor II\\data\\test\\infected\\img_0_1836.jpg")

predictimage("D:\\3rd Year\\Sem 6\\Minor II\\data\\test\\notinfected\\img_0_126.jpg")