from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py

TRAIN_DIR = 'C:\\Users\\KIIT\\Desktop\\Code Clause Projects\\Task 3\\images\\train_resized'
TEST_DIR = 'C:\\Users\\KIIT\\Desktop\\Code Clause Projects\\Task 3\\images\\test_resized'
TRAIN_CSV = 'C:\\Users\\KIIT\\Desktop\\Code Clause Projects\\Task 3\\labels\\trainLabels15.csv'
TEST_CSV = 'C:\\Users\\KIIT\\Desktop\\Code Clause Projects\\Task 3\\labels\\testLabels15.csv'

train_labels_df = pd.read_csv(TRAIN_CSV)
test_labels_df = pd.read_csv(TEST_CSV)

def createdataframe(dir, labels_df):
    image_paths = []
    labels = []
    for index, row in labels_df.iterrows():
        image_name = row['image']
        label = row['level']
        image_paths.append(os.path.join(dir, image_name + '.jpg'))
        labels.append(label)
    return image_paths, labels

train_images, train_labels = createdataframe(TRAIN_DIR, train_labels_df)
test_images, test_labels = createdataframe(TEST_DIR, test_labels_df)

# Define a function to load and preprocess images
def preprocess_images(image_paths, target_size=(224, 224)):
    images = []
    for image_path in tqdm(image_paths):
        img = load_img(image_path, color_mode='grayscale', target_size=target_size)
        img = img_to_array(img)
        img = img / 255.0  # Normalize pixel values to be between 0 and 1
        images.append(img)
    return np.array(images)

# Load and preprocess train and test images
x_train = preprocess_images(train_images)
x_test = preprocess_images(test_images)

le = LabelEncoder()
y_train = le.fit_transform(train_labels)
y_test = le.transform(test_labels)

y_train = to_categorical(y_train, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)

# Define the model
model = Sequential()

# Adding Convolutional Layers
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test))

# Save the model
model.save("Blindness_Detection.h5")

# Save the label encoding information
np.save('label_encoding.npy', le.classes_)

from keras.models import model_from_json

model_json = model.to_json()
with open("Blindness_Detection.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights("Blindness_Detection.h5")

from keras.models import model_from_json

# Load the model architecture from JSON
with open("Blindness_Detection.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)

# Load the model weights into the loaded model
loaded_model.load_weights("Blindness_Detection.h5")

model.save("Blindness_Detection.h5")

label = ['No_DR','Mild','Moderate','Severe','Proliferate_DR']

import cv2

def ef(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1, 224, 224, 1)
    return img / 255.0

image_path4 = 'images/test_images/e7d2c2c3b30f.png'
image_path3 = 'images/test_images/e93394175a19.png'
image_path2 = 'images/test_images/f61bf44c677c.png'
image_path1 = 'images/test_images/fb6b8200b7f8.png'
image_path0 = 'images/test_images/fcc32dffd24d.png'

print("The original image is of label 4")
img4 = ef(image_path4)
predicted_label4 = model.predict(img4)
print("Predicted Label:", predicted_label4)
print("Predicted Label:",predicted_label4.argmax())

print("The original image is of label 3")
img3 = ef(image_path3)
predicted_label3 = model.predict(img3)
print("Predicted Label:", predicted_label3)
print("Predicted Label:",predicted_label3.argmax())

print("The original image is of label 2")
img2 = ef(image_path2)
predicted_label2 = model.predict(img2)
print("Predicted Label:", predicted_label2)
print("Predicted Label:",predicted_label2.argmax())

print("The original image is of label 1")
img1 = ef(image_path1)
predicted_label1 = model.predict(img1)
print("Predicted Label:", predicted_label1)
print("Predicted Label:",predicted_label1.argmax())

print("The original image is of label 0")
img0 = ef(image_path0)
predicted_label0 = model.predict(img0)
print("Predicted Label:", predicted_label0)
print("Predicted Label:",predicted_label0.argmax())