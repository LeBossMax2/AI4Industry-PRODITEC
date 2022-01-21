import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from sklearn.utils import class_weight

from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Recall

import pathlib
import shutil

batch_size = 128
learning_rate = 0.0001
epochs = 20

data_dir = "data/IA_test" # modifier data pour n'avoir que deux categorie/fichier
data_dir = pathlib.Path(data_dir)

crop = layers.Cropping2D(cropping=80)
resize = layers.Resizing(224, 224)
rescaling = layers.Rescaling(1/255, offset=0)

def preprocess_image(img, label):
    #img = crop(img)
    img = resize(img)
    img = rescaling(img)
    return img, label

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(258, 258),
  crop_to_aspect_ratio=True,
  batch_size=batch_size)

class_names = train_ds.class_names
train_ds = train_ds.map(preprocess_image)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(258, 258),
  crop_to_aspect_ratio=True,
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

def positive_accuracy(true_label, pred):
    true_pred = K.round(tf.boolean_mask(pred, tf.math.less(true_label, tf.constant([3.0])), axis = 0))
    return tf.cond(
        tf.size(true_pred) == 0,
        lambda: tf.constant([1.]),
        lambda: K.mean(tf.math.less(true_pred, tf.constant([3.0])))
    )

def negative_accuracy(true_label, pred):
    false_pred = K.round(tf.boolean_mask(pred, tf.math.greater_equal(true_label, tf.constant([3.0])), axis = 0))
    return tf.cond(
        tf.size(false_pred) == 0,
        lambda: tf.constant([1.]),
        lambda: K.mean(tf.math.greater_equal(false_pred, tf.constant([3.0])))
    )

def build_model():

    input = layers.Input(shape=(224, 224, 3))

    resnet = keras.applications.ResNet50V2(include_top=False, input_shape=(224, 224, 3))
    x = resnet(input, training=True)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=input, outputs=x)

    model.compile(optimizer=Adam(learning_rate),
                  loss="mse",
                  metrics=['accuracy', positive_accuracy, negative_accuracy])
    return model

def train():

    model = build_model()

    y_train = np.concatenate([y.numpy() for x, y in val_ds])
    print(y_train)
    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.arange(0, num_classes), y = y_train)
    class_weights = dict(zip(np.arange(0, num_classes), class_weights))

    print(class_weights)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        class_weight=class_weights,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=8, restore_best_weights=True)]
    )

    plt.figure(figsize=(15,5))
    for i, t in enumerate(["loss", "accuracy", "positive_accuracy", "negative_accuracy"]):
        plt.subplot(1, 4, i + 1)
        plt.plot(history.history[t], label=t+" train")
        plt.plot(history.history["val_" + t], label=t+" val")
        plt.title('model ' + t)
        plt.ylabel(t)
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
    plt.savefig("history.png")

    print('\nSaving...')
    model.save_weights("My_model")
    print('Ok')

def evaluate():

    model = build_model()
    model.load_weights("My_model")

    val_eval = model.evaluate(val_ds)
    print("Val:", val_eval)

    val_pred = model.predict(val_ds)
    labels = tf.concat([y for x, y in val_ds], axis=0)

    mat = tf.math.confusion_matrix(labels, K.round(K.clip(val_pred, 0, 4)), num_classes=num_classes)

    print(mat)
    
    tf.print(K.sum(tf.linalg.diag_part(mat)) / K.sum(mat), K.sum(mat[:3, :3]) / K.sum(mat[:3, :]), K.sum(mat[3:, 3:]) / K.sum(mat[3:, :]))

def predict(data_test, cat) :

    model = build_model()
    model.load_weights("My_model")

    img_array = tf.expand_dims(data_test, 0)

    predictions = model.predict(img_array)
    score = predictions[0]

    print("The score of this image is", score, ", actuel category", cat)

train()

evaluate()

predict(next(iter(val_ds))[0][0], next(iter(val_ds))[1][0])