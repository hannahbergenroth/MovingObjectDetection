# import the necessary packages
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import vgg16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import os
import cv2

ROOT_PATH = "training_data"
WEIGHTS_PATH_NO_TOP = "model/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
CHECKPOINT_WEIGHTS = "model/weights/tl_model_v1.weights.best.hdf5"
CHECKPOINT_PATH = "model/my_model"
EPOCHS = 50
BATCH_SIZE = 32

print("[INFO] loading images...")
imagePaths = list(paths.list_images(ROOT_PATH))
data = []
labels = []

amount = 2056

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32))

    data.append(image)
    labels.append(label)


# convert the data and labels to NumPy arrays
x_train = np.array(data)
y_train = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 80% of the data for training and the remaining 20% for
# testing
(trainX, testX, trainY, testY) = train_test_split(x_train, labels, test_size=0.20, stratify=labels, random_state=42,
                                                  shuffle=True)

print("train: {0} | test: {1}".format(trainX.shape[0], testX.shape[0]))

# Initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Initialize the validation/testing data augmentation object
valAug = ImageDataGenerator()

# Normalize image data to 0-to-1 range
# x_train = vgg16.preprocess_input(x_train)

# define the imagenet mean subtraction (in RGB order) and set the
# mean subtraction value for each of the data augmentation objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean


# Load the VGG16 network, ensuring the head FC layer sets are left off
def vgg_16(weights_path=None):
    model = Sequential()
    model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Load imagenet weights
    if weights_path == 'imagenet':
        model.load_weights(WEIGHTS_PATH_NO_TOP)

    # Freeze layers during training
    for layer in model.layers[:-2]:
        layer.trainable = False

    return model


baseModel = vgg_16('imagenet')

# Construct the head of the model that will be placed on top of base model
headModel = baseModel.output
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(256, activation='relu')(headModel)
headModel = Dropout(0.2)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Head FC on top of base model creates the new model
newModel = tensorflow.keras.models.Model(inputs=baseModel.input, outputs=headModel)

print("[INFO] summary for base model...")
print(baseModel.summary())

print("[INFO] compiling model...")
# Stochastic gradient descent optimizer
opt = tensorflow.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9, decay=1e-7)
# Adam optimizer
opt2 = tensorflow.keras.optimizers.Adam(learning_rate=0.00001)
newModel.compile(loss="categorical_crossentropy", optimizer=opt2, metrics=["accuracy"])

bestModel = ModelCheckpoint(CHECKPOINT_PATH,
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True,
                            verbose=1,
                            save_weights_only=False)

earlyStopping = EarlyStopping(monitor='val_loss', patience=8)

print("[INFO] training...")
H = newModel.fit(
    x=trainAug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    validation_data=valAug.flow(testX, testY),
    validation_steps=len(testX) // BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=True,
    callbacks=[earlyStopping, bestModel]
)

print('Test accuracy: {0:.3}%'.format(newModel.evaluate(x=testX, y=testY, batch_size=BATCH_SIZE)[1] * 100))

# Evaluate the network
print("[INFO] evaluating network...")
predictions = newModel.predict(x=testX.astype("float32"), batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# Plot the training loss and accuracy
N = len(H.history["loss"])
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Model Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("plots/loss.png")
plt.show()
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Model Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("plots/accuracy.png")
plt.show()
plt.clf()

# Compute confusion matrix
cnf_matrix = pd.crosstab(lb.classes_[testY.argmax(axis=1)], lb.classes_[predictions.argmax(axis=1)])
fig, ax = plt.subplots(figsize=(5, 5))

sns = sns.heatmap(cnf_matrix, linewidths=1, annot=True, ax=ax, cmap='Blues', fmt='g')
plt.xlabel("Actual class")
plt.ylabel("Predicted class")
plt.savefig("plots/confusion_matrix.png")
plt.show()
