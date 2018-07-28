# **Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
* 
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the powerful NVIDIA Convolution Neural Network for this project.The below figure depicts the architecture and layers of the model.


##### NVIDIA Architecture


![NVIDIA Architecture](/Writeup_Images/NVIDIA_Architecture.png)

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting, the data was split into 80% training and  20% validation data.The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 

I plotted the models mean squared error loss and found that it was low for training set but high on validation set and hence the model was overfitting.Hence I added Dropout to the Dense layers.

Below plot signifies the impact of adding Dropout to the model.


##### Model MSE with Dropout


![Model MSE with Dropout](/Writeup_Images/loss_plot_with_dropout.png)


##### Model MSE without Dropout


![Model MSE without Dropout](/Writeup_Images/loss_plot_without_dropout.png)

#### 3. Model parameter tuning

Adam Optimizer with default learning rate was used.  

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used data from all three cameras -  center, left and right.
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


##### Creation of Training and Validation Data

The project setup consists of using three cameras - left, center and right to collect the training data while driving.I used the data set given by Udacity.Data is everything in Machine Learning.Data quantity and data quality are extremely important when it comes to training the model.The Udacity dataset comprises of data from three different cameras -left, right and center.I first started with center camera data.Its important to analyze the training data set in order to understand the type of preprocessing steps required prior training.The first step was to convert the images to RGB since the drive.py file loads images in RGB to predict the steering angles. Since I was using the Udacity dataset, the number of images were limited and hence used data augmentation method like flipping to increase the number of images.Since the image was flipped the steering angle was negated.

The next step was to use the images from left and right camera.This step helps us train the model as if the images from the left and right camera are coming from the center camera.This teaches the car how to steer itself back to the center of the road if its drifts off to left or right of the road which happens during left and right turns. Inorder to do this, we need to add/subtract correction or adjustment angle to the steering angle.After few experimentation,0.1 worked best for the model.

Total Training Data : 38572
Total Validation data : 9644
Total Images : 48216


##### Original Center Camera Image


![Center Original](/Writeup_Images/center_orig.jpg) 
     
##### Flipped Center Camera Image


![Center Flipped](/Writeup_Images/center_flipped.jpg) 

##### Original Left Camera Image


![Left Original](/Writeup_Images/left_orig.jpg)        

##### Flipped Left Camera Image


 ![Left Flipped](/Writeup_Images/left_flipped.jpg) 

##### Original Right Camera Image


![Right Original](/Writeup_Images/right_orig.jpg)        

##### Flipped Right Camera Image


![Right Flipped](/Writeup_Images/right_flipped.jpg) 

##### Model Architecture

The overall strategy for deriving a model architecture was to start with the default NVIDIA architecture.This model gave better results compared to LENET.I added Lambda layer to mean centering and normalization.Also added a layer to crop 70 pixel from top and 25 from bottom.This was required since the data  in top was mainly trees and sky while data in bottom was hood of the car.Cropping helps in extracting data from Region of interest.

##### Uncropped Original Image                                ##### Cropped Image


![Center Original](/Writeup_Images/center_orig.jpg)          ![Center Cropped](/Writeup_Images/center_cropped.jpg)
  



In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Hence I added Dropout of 0.5 to the Dense Layers.This step reduced the MSE for validation set.We use MSE instead of Cross Entropy as we need a regression network and not a classification network.Adam optimizer was used.Initially I trained the model for 10 Epochs and noticed that the MSE spiked after Epoch 4-5.Hence the final Epoch number used was 5 with Dropout of 0.5.This gave perfect result with no overfitting.The model was saved as Model.h5

The final step was to run the simulator with the saved model to see how well the car was driving.I first started with the default speed of 9mph.The car did pretty good sticking to the center of the road.


###### 9mph Video


![9mph Video](/Writeup_Images/video_9mph.gif) 


The next step I wanted to try was increasing the speed to see how the car handles itself while taking steep turns, especially the one after the bridge.The below video is simulation with speed 15mph.I have checked in the drive.py file with speed 15mph.


###### 15mph Video


![15mph Video](/Writeup_Images/video_15mph.gif) 



#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

| Layer (type)             | Output Shape        | Param # |
|:------------------------:|:-------------------:|:-------:|
|lambda_1 (Lambda)         | (None, 160, 320, 3) | 0       |
|cropping2d_1 (Cropping2D) | (None, 65, 320, 3)  | 0       |
|conv2d_1 (Conv2D)         | (None, 31, 158, 24) | 1824    |
|conv2d_2 (Conv2D)         | (None, 14, 77, 36)  | 21636   |
|conv2d_3 (Conv2D)         | (None, 5, 37, 48)   | 43248   |
|conv2d_4 (Conv2D)         | (None, 3, 35, 64)   | 27712   |
|conv2d_5 (Conv2D)         | (None, 1, 33, 64)   | 36928   |
|flatten_1 (Flatten)       | (None, 2112)        | 0       |
|dense_1 (Dense)           | (None, 100)         | 211300  |
|dropout_1 (Dropout)       | (None, 100)         | 0       |
|dense_2 (Dense)           | (None, 50)          | 5050    |
|dropout_2 (Dropout)       | (None, 50)          | 0       |
|dense_3 (Dense)           | (None, 10)          | 510     |
|dropout_3 (Dropout)       | (None, 10)          | 0       |
|dense_4 (Dense)           | (None, 1)           | 11      |


Optimizer : Adam
Loss Function : Mean Squared Error
Epoch : 5
Steering Adjustment : 0.1
Speed : 15mph