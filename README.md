# CarND-Behavioral-Cloning-Project
My submission for the Udacity Self-Driving Car Nanodegree program Project 3 - Behavioral Cloning
The goals / steps of this project are the following:

Use the simulator to collect data of good driving behavior
Build, a convolution neural network in Keras that predicts steering angles from images
Train and validate the model with a training and validation set
Test that the model successfully drives around track one without leaving the road
Summarize the results with a written report



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I chose to use the [Nvidia's End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) model. I diverged by passing cropped camera images as RGB, and not YUV, with adjusting brightness and by using the steering angle as is.

I also experimented with [comma.ai](http://comma.ai/), [Steering angle prediction model](https://github.com/commaai/research/blob/master/train_steering_model.py). However, I didn't notice a better result with the same training data and decided to stick to NVIDIA model and improve my training data. 

The model represented here is what I used, it is an implementation of the nvidia model. It is developed in python using keras (High Level Tensorflow library) in model.py and returned from the `build_nvidia_model` method. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


The training of the neural net model, is achieved with driving behaviour data captured, in training mode, within the simulator itself. Additional preprocessing occurs as part of batch generation of data for the neural net training.

##Model Architecture


###Input

The input is 66x200xC with C = 3 RGB color channels.

###Architecture
**Layer 0: Normalisation** to range -1, 1 (1./127.5 -1)

**Layer 1: Convolution** with strides=(2,2), valid padding, kernel 5x5 and output shape 31x98x24, with **elu activation** and **dropout**

**Layer 2: Convolution** with strides=(2,2), valid padding, kernel 5x5 and output shape 14x47x36, with **elu activation** and **dropout**

**Layer 3: Convolution** with strides=(2,2), valid padding, kernel 5x5 and output shape 5x22x48, with **elu activation** and **dropout**

**Layer 4: Convolution** with strides=(1,1), valid padding, kernel 3x3 and output shape 3x20x64, with **elu activation** and **dropout**

**Layer 5: Convolution** with strides=(1,1), valid padding, kernel 3x3 and output shape 1x18x64, with **elu activation** and **dropout**

**flatten** 1152 output

**Layer 6: Fully Connected** with 100 outputs and **dropout**

**Layer 7: Fully Connected** with 50 outputs and **dropout**

**Layer 8: Fully Connected** with 10 outputs and **dropout**

dropout was set aggressively on each layer at .25 to avoid overtraining
###Output

**Layer Fully Connected** with 1 output value for the steering angle.

### Visualisation
[Keras output plot](./model.png)

## Data preprocessing and Augmentation
The simulator captures data into a csv log file which references left, centre and right captured images within a sub directory. Telemetry data for steering, throttle, brake and speed is also contained in the log. Only steering was used in this project.


Before being fed into the model, the images are cropped to 66x200 starting at height 60 with width centered -


As seen in the following histogram a significant proportion of the data is for driving straight and its lopsided to left turns (being a negative steering angle is left) when using data generated following my conservative driving laps.
![Steering Angle Histogram](https://raw.githubusercontent.com/hortovanyi/udacity-behavioral-cloning-project/master/images/steering_histogram.png)

The log file was preprocessed to remove contiguous rows with a history of >5 records, with a 0.0 steering angle. This was the only preprocessing done outside of the batch generators used in training (random rows are augmented/jittered for each batch at model training time).

A left, centre or right camera was selected randomly for each row, with .25 angle (+ for left and - for right) applied to the steering.

Jittering was applied per [Vivek Yadav's post ](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0) to augment data. Images were randomly transformed in the x range by 100 pixels and in the y range by 10 pixels with 0.4 per xpixel adjusted against the steering angle. Brightness via a HSV (V channel) transform (.25 + a random number in range 0 to 1) was also performed.

During batch generation, to compensate for the left turning, 50% of images were flipped (including reversing steering angle) if the absolute steering angle was > .1.

Finally images are cropped per above before being batched.

### Model Training

Data was captured from the simulator. I drove conservatively around the track three times paying particular attention to the sharp right turn. I found connecting a PS3 controller allowed finer control then using the keyboard. At least once I waited till the last moment before taking the turn. This seems to have stopped the car ending up in the lake. Its also helped to overcome a symptom of the bias in the training data towards left turns. To further offset this risk, I validated the training using a test set I'd captured from the second track, which is a lot more windy.


The Adam Optimizer was used with a mean squared error loss. A number of hyper-parameters were passed on the command line. The command I used looks such for a batch size of 500, 10 epochs (dropped out early if loss wasn't improving), dropout at .25 with a training size of 50000 randomly augmented features with adjusted labels and 2000 random features & labels used for validation

```
python model.py --batch_size=500 --training_log_path=./data --validation_log_path=./datat2 --epochs 10 \
--training_size 50000 --validation_size 2000 --dropout .25
```


### Model Testing
To meet requirements, and hence pass the assignment, the vehicle has to drive around the first track staying on the road and not going up on the curb.

The model trained (which is saved), is used again in testing. The simulator feeds you the centre camera image, along with steering and throttle telemetry. In response you have to return the new steering angle and throttle values. I hard coded the throttle to .35. The image was cropped, the same as for training, then fed into the model for prediction giving the steering angle.

```python

steering_angle = float(model.predict(transformed_image_array, batch_size=1))
throttle = 0.35
```
