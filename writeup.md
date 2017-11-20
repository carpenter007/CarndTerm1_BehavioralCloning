**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./documentation/model_base_architecture.png "Model Visualization"
[image2]: ./documentation/flipped_image.png "Flipped image"
[image3]: ./documentation/left_centre_right.png "camera views"
[image4]: ./documentation/cropped_image.png "cropped image"

---
## Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* augmentation.py for creating new training data out of the recorded data
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 which shows how the car drives through track 1 with the given model
* visualization.py to visualize some training data

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

The overall strategy for deriving a model architecture was to use a deep neural network with trained weights to minimize the mean squared error between the steering output by the network and the output by some recorded datasets with the simulator.

My first step was to use a convolution neural network model similar to the network NVIDIA model used in the DAVE-2 system. DAVE-2 is a robust system for driving autonomously on public roads. Therefore I thought this model might be appropriate. The shape of the DAVE-2 input images is 3@66x200 which is quite comparable with the simulators image shape of 3@160x320 (or after cropping: 3@65x320).

![alt text][image1]

To combat the overfitting, I modified the model so that there is a dropout layer (0.25 dropout) after the model is flatted.
Further I applied Keras L2 weights regularizer between the convolution layers.

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 59-72) 

The model includes RELU layers to introduce nonlinearity (code line 62 - 66), and the data is normalized in the model using a Keras lambda layer (code line 61). 
The model contains a dropout layer in order to reduce overfitting (model.py lines 68). The dropout layer is placed just after the flatten.

The final model architecture (model.py lines 59-72) consisted of a convolution neural network with the following layers and layer sizes

- input image with 3@160x320
- normalization layer: (pixels / 255) - 0.5 and cropping image (70pixel and 25 pixel from the top and the bottom of the image)
- convolutional feature map with 5x5 filter
- Weight regularizer (0.01)
- convolutional feature map with 5x5 filter
- Weight regularizer (0.01)
- convolutional feature map with 5x5 filter
- Weight regularizer (0.01)
- convolutional feature map with 3x3 filter
- convolutional feature map with 3x3 filter
- flatten layer
- Dropout with a keep rate of 0.75
- fully-connected layer of 100
- fully-connected layer of 50
- fully-connected layer of 10
- fully-connected layer of 1


The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
For details about how I created the training data, see the next section. 

## Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the images of the centre camera lane driving. Further I used the left and right camera pictures and fitted their corresponding angle values. As suggested, the correction factor is 3 angles in either direction ( 1.0/25 * 3 - the car has a steering angle of -25 to 25 in the simulation and it is normalized in the driving_log.csv file from -1 to 1, so 3 angles in either direction are 0.12 or -0.12.  This helped to get the car back to the centre of the street. To get the steering behaviour more smoothly, I decided to generate more camera data which include different situations.
![alt text][image3]

To train the model only with relevant areas of the pictures, I cropped the images at the top and at the bottom.
![alt text][image4]

The training data consists of:
- two laps driving just as good as possible through the complete track.
- one lap driving the track clockwise reversed
- additional driving every curve very slow but as smoothly as possible
- driving one lap which shows how to react after the found itself to close to the side of the road
- finally I made a flipped copy of all centre image data inclusive the steering values

![alt text][image2]

After the collection and augmentation process, I had 57500 images and steering angles to train and validate the model.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 


## Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 67).
I used this training data for training the model. The validation set helped to determine if the model was over or under fitting. The ideal number of epochs is generated by using Keras callback functions 'early stopping' and 'model checkpoints'. This helps me to stop the training before the model is overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Thanks to Udacity for this great project. I give thanks especially to my mentor Rajesh and my reviewer, who gave me very good suggestions to improve the project.
