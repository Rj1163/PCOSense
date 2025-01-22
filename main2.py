import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(root_dir):
    """
    Load the number of images in each directory.
    """
    number_of_images = {}
    for dir in os.listdir(root_dir):
        number_of_images[dir] = len(os.listdir(os.path.join(root_dir, dir)))
        print("", dir, "", number_of_images[dir])
    return number_of_images

def preprocess_image(path):
    """
    Preprocess images using DenseNet's preprocess_input function.
    """
    image_data = ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input)
    image_generator = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='binary')
    return image_generator

def build_model():
    """
    Build the DenseNet121 model with custom dense layers for classification.
    """
    base_model = DenseNet121(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model

def train_model(model, train_data, val_data):
    """
    Train the model.
    """
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    mc = tf.keras.callbacks.ModelCheckpoint(filepath="bestmodel_densenet.keras",
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max')
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                           min_delta=0.01,
                                           patience=5,
                                           verbose=1,
                                           mode='max')
    callbacks = [mc, es]

    history = model.fit(train_data,
                        steps_per_epoch=len(train_data),
                        epochs=30,
                        validation_data=val_data,
                        validation_steps=len(val_data),
                        callbacks=callbacks)
    return history

def evaluate_model(model, test_data):
    """
    Evaluate the model on the test data.
    """
    test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
    print("Test Accuracy:", test_accuracy)

def predict_image(model, path):
    """
    Predict the class of an image.
    """
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    prediction = model.predict(img_array)
    if prediction[0] >= 0.5:
        print("Not Affected")
    else:
        print("Affected")
    plt.imshow(img)
    plt.title("Input Image")
    plt.show()

if __name__ == "__main__":
    ROOT_DIR = "D:\\3rd Year\\Sem 6\\Minor II\\data"
    number_of_images = load_data(ROOT_DIR)

    train_path = os.path.join(ROOT_DIR, "train")
    test_path = os.path.join(ROOT_DIR, "test")
    val_path = os.path.join(ROOT_DIR, "validation")

    train_data = preprocess_image(train_path)
    test_data = preprocess_image(test_path)
    val_data = preprocess_image(val_path)

    model = build_model()
    history = train_model(model, train_data, val_data)
    evaluate_model(model, test_data)

    # Example predictions
    predict_image(model, "D:\\3rd Year\\Sem 6\\Minor II\\data\\test\\infected\\img_0_1836.jpg")
    predict_image(model, "D:\\3rd Year\\Sem 6\\Minor II\\data\\test\\notinfected\\img_0_126.jpg")

import joblib
import tensorflow as tf

# Save the model as .h5 file
model.save("best_model_densenet.h5")

# Load the model
loaded_model = tf.keras.models.load_model("best_model_densenet.h5")

# Save the loaded model as .pkl file
joblib.dump(loaded_model, "best_model_densenet.pkl")