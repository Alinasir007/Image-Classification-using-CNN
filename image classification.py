from keras.datasets import cifar10
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
(x_train, y_train), (x_test, y_test)= cifar10.load_data()
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

#get the shapes
print('x_train shape :', x_train.shape)
print('y_train shape :', y_train.shape)
print('x_test shape :', x_test.shape)
print('y_test shape :', y_test.shape)

#first image
print(x_train[0])

#show image
img= plt.imshow(x_train[0])
plt.show(img)

#The alternative code for showing image
'''cv2.imshow('one', x_train[0])
cv2.waitKey(5000)
cv2.destroyAllWindows()'''

print("The label is :", y_train[0])

#one hot-encoding: to convert the labels into number for neural network
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#print the new labels
print(y_train_one_hot[0])
print(y_train_one_hot)
print(f"Rows in train_set: {len(y_train_one_hot)}")

#normalize the value between 0-255
x_train= x_train/255
x_test= x_test/255

#Build the CNN network
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

#Create the architecture
model= Sequential()

#Convolution Layer
model.add(Conv2D(32, (5,5), activation= 'relu', input_shape= (32,32,3)))

#MaxPooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

#Convolution Layer
model.add(Conv2D(32, (5,5), activation= 'relu'))

#MaxPooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

#flattening
model.add(Flatten())

#dense
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

#compile the model
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train the model
hist= model.fit(x_train, y_train_one_hot, batch_size=256, epochs= 10, validation_split=0.3)

#accuracy test
print(model.evaluate(x_test, y_test_one_hot)[1])

#visualize the model accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'val'], loc='upper left')
plt.show()

#visualize the loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'val'], loc='upper right')
plt.show()

#load the data
image= Image.open('C:\\Users\\admin\\PycharmProjects\\image\\bird.JPG')
np_image= np.array(image)
print( np_image.shape)
imge= plt.imshow(np_image)
plt.show(imge)

from skimage.transform import resize
image_resized= resize(np_image, (32,32,3))
print(image_resized.shape)
image_change = plt.imshow(image_resized)
plt.show(image_change)

#Getting probabilities for each class
probabilities= model.predict(np.array([image_resized,]))
print(probabilities)


#labelling correctly
number_to_classes= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'frog', 'horse', 'sheep', 'truck']
index= np.argsort(probabilities[0,:])
print('It most likely belongs to class:', number_to_classes[index[9]], '--with a probability of:', probabilities[0, index[9]])



