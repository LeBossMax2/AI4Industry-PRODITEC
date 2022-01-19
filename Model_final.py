import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Recall

import pathlib
import shutil

batch_size = 64 # TODO tester un batch size plus gros
learning_rate = 0.0002
epochs = 20

data_dir = "data/IA" # modifier data pour n'avoir que deux categorie/fichier
data_dir = pathlib.Path(data_dir)

crop = layers.Cropping2D(cropping=80)
resize = layers.Resizing(224, 224)
rescaling = layers.Rescaling(1/255, offset=0)

def preprocess_image(img, label):
    img = crop(img)
    img = resize(img)
    img = rescaling(img)
    return img, label

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(480, 480),
  crop_to_aspect_ratio=True,
  label_mode='categorical',
  batch_size=batch_size)

class_names = train_ds.class_names
train_ds = train_ds.map(preprocess_image)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(480, 480),
  crop_to_aspect_ratio=True,
  label_mode='categorical',
  batch_size=batch_size).map(preprocess_image)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(64):
    ax = plt.subplot(8, 8, i + 1)
    plt.imshow(images[i].numpy())
    plt.axis("off")
plt.savefig("test.png")

AUTOTUNE = tf.data.AUTOTUNE

#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

def categorical_mse(true_label, pred):
    true_cat = K.argmax(true_label)
    pred_cat = K.argmax(pred)
    
    return K.mean((true_cat - pred_cat)**2)

def build_model():
    model = keras.applications.ResNet50V2(weights=None, include_top=True, classes=num_classes, classifier_activation='softmax')

    model.compile(optimizer=Adam(learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy', Recall(), categorical_mse])
    return model

def train():

    model = build_model()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=8, restore_best_weights=True)]
    )

    print('\nSaving...')
    model.save_weights("My_model")
    print('Ok')

def evaluate():

    model = build_model()
    model.load_weights("My_model")

    val_eval = model.evaluate(val_ds)
    print("Val:", val_eval)

def predict(data_test) :

    model = build_model()
    model.load_weights("My_model")

    img_array = tf.expand_dims(data_test, 0)

    predictions = model.predict(img_array)
    score = predictions[0]

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

train()

evaluate()

predict(next(iter(val_ds))[0][0])