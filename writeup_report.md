# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[learning_rate]: ./examples/learning_rate.png "Learning Rate"
[mse_loss]: ./examples/mse_loss_over_epoch.png "MSE Loss"
[model]: ./examples/model.png "Model Architecture"
[speed_graph1]: ./examples/speed_graph_track1.png "Speed Graph Track 1"
[speed_graph2]: ./examples/speed_graph_track2.png "Speed Graph Track 2"
[center]: ./examples/training-center.jpg "Center"
[left]: ./examples/training-left.jpg "Left"
[right]: ./examples/training-right.jpg "Right"

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
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

My model consists of 5 layers of convolution neural network and 4 fully connected layers as the adoption of NVIDIA Self Driving Cars Model (code line 119 - 149)

The model includes RELU operations to introduce nonlinearity (code line 132-137), and the data is normalized in the model using a Keras lambda layer (code line 126). On top of that, I applied 2D cropping in order to make the model only see the road (code line 129) 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 140, 142, 144, and 146). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 172). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 219). However, since the training didn't affect val_loss after 5 epochs, I applied LearningRateScheduler from Keras and applied exponential decay (code line 199). On top of that, I use checkpoint to save the best model (code line 206-212). This allows me to ensure that my model is right fit, not over fit or under fit.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving to the opposite direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the image data to predict the steering angle.

My first step was to use a convolution neural network model similar to the NVIDIA Architecture. I thought this model might be appropriate because they use it for self driving cars. Five convolutional layers allow the model to extract mode information from the image and four fully connected layers will calculate the output of the steering angle needed to drive the car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding 4 additional dropout operations after each dense layer. The dropout rate was set to 0.5 for training.

Then I augmented the data by flipping it, normalize all input and make it has zero variance. After that, to improve the model, I crop the image to see only the road.
Using 3 cameras instead of one camera from the center also helps to improve the model perform better. I added angle correction for left and right cameras to compensate the position (code line 61-74)

The final step was to run the simulator to see how well the car was driving around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 119-148).

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is some examples of center lane driving from 3 cameras on the car:

Left camera:

![alt text][left]

Center camera:

![alt text][center] 

Right camera:

![alt text][right]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to the center lane when it goes off track.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would make the model perform better in general.

After the collection process, I had 93003 number of data points for both tracks. I then preprocessed this data by normalization, applying zero mean variance, and cropping unnecessary part of the image.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as I used LearningRateScheduler to optimize the learning rate.
![alt text][learning_rate]

Below is the training history.

![alt text][mse_loss]

After I tested, the model is good enough to let the car travel on both tracks smoothly. I modified the PI value and the desired speed in both tracks to optimize the result. The speed was set to 15 for track 1 and 12 for track 2 due to its extreme condition. 

Below is the speed visualization for track 1 from 0 to its desired speed.
![alt text][speed_graph1]

And below is the screenshot when running on track 2 which is much more difficult compared to track 1.
![alt text][speed_graph2]