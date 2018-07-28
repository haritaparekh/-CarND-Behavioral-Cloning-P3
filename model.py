import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D,Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
import matplotlib.pyplot as plt

def preprocess(image,flip):                              #Preprocessing steps like convert to RGB and flipping images
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if flip is True:
        image_flipped = np.fliplr(RGB_image)
        return image_flipped
    else:
        return RGB_image

lines = []
images = []
measurements = []
steering_adjust = 0.1
with open ('data/driving_log.csv') as csvfile:  #Reading CSV file
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
        
for line in lines:
    for i in range(3):
        image_path = 'data/IMG/' + line[i].split('/')[-1]            #Extracting Image path from CSV file
        image = cv2.imread(image_path)
        if i == 0:                                                   #Loop for Center Camera Images
            images.append(preprocess(image,False)) 
            measurements.append(float(line[3]))
            images.append(preprocess(image,True))
            measurements.append(-float(line[3]))
        if i == 1:                                                    #Loop for Left Camera Images
            images.append(preprocess(image,False))
            measurements.append(float(line[3]) + steering_adjust)     #Making steering angle adjustments
            images.append(preprocess(image,True))
            measurements.append(-(float(line[3]) + steering_adjust))
        if i == 2:                                                     #Loop for Right Camera Images
            images.append(preprocess(image,False))
            measurements.append(float(line[3]) - steering_adjust)       #Making steering angle adjustments
            images.append(preprocess(image,True))
            measurements.append(-(float(line[3]) - steering_adjust))
               

X_train = np.array(images)                     #Array of all images
Y_train = np.array(measurements)               #Array of all steering angles

#Convolution Neural Network

model = Sequential()                           
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), activation='relu',strides=(2,2)))
model.add(Conv2D(36, (5, 5), activation='relu',strides=(2,2)))
model.add(Conv2D(48, (5, 5), activation='relu' ,strides=(2,2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train,Y_train,validation_split=0.2,shuffle=True,epochs=5)

 

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss_plot_with_dropout.png')


# Save the model
model.save('model.h5')
exit()    
    