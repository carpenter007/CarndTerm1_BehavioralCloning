**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./documentation/model_base_architecture.png "Model Visualization"

---
## Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* augmentation.py for creating new training data out of the recorded data
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 52-65) 

The model includes RELU layers to introduce nonlinearity (code line 55 - 59), and the data is normalized in the model using a Keras lambda layer (code line 53). 

The model contains a dropout layer in order to reduce overfitting (model.py lines 61). The dropout layer is placed just after the flatten.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

## Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 67).

## Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the images of the centre camera lane driving. First I used the left and right camera pictures and fitted their corresponding angle values between 0.1 and 0.4. This helped to get the car back to the centre of the street. But it also ended up with a not smoothly behaviour. So I decided to generate just more centre camera data which include different situations.

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

The overall strategy for deriving a model architecture was to use a deep neural network with trained weights to minimize the mean squared error between the steering output by the network and the output by some recorded datasets with the simulator.

My first step was to use a convolution neural network model similar to the network NVIDIA model used in the DAVE-2 system. DAVE-2 is a robust system for driving on public roads. Therefore I thought this model might be appropriate. The shape of the DAVE-2 input images is 3@66x200 which is quite comparable with the simulators image shape of 3@160x320 (or after cropping: 3@65x320).

To combat the overfitting, I modified the model so that there is a dropout layer (0.25 dropout) after the model is flatted.

Then I prepared the training data. The training data consists of:
- two laps driving just as good as possible through the complete track.
- one lap driving the track clockwise reversed
- additional driving every curve very slow but as smoothly as possible
- driving one lap which shows how to react after the found itself to close to the side of the road
- finally I made a flipped copy of all the data inclusive the steering values

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

The final model architecture (model.py lines 52-65) consisted of a convolution neural network with the following layers and layer sizes

![alt text][image1]

So we have
- input image with 3@160x320
- normalization layer: (pixels / 255) - 0.5 and cropping image (70pixel and 25 pixel from the top and the bottom of the image)
- convolutional feature map with 5x5 filter
- convolutional feature map with 5x5 filter
- convolutional feature map with 5x5 filter
- convolutional feature map with 3x3 filter
- convolutional feature map with 3x3 filter
- flatten layer
- fully-connected layer of 100
- fully-connected layer of 50
- fully-connected layer of 10
- fully-connected layer of 1


After the collection and augmentation process, I had 28783 number of data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4. More epochs only results into better training rates but not in better validation rates. I used an adam optimizer so that manually training the learning rate wasn't necessary.

