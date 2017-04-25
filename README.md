# CarND-Behavioral-Cloning-Project
My submission for the Udacity Self-Driving Car Nanodegree program Project 3 - Behavioral Cloning
The goals / steps of this project are the following:

Use the simulator to collect data of good driving behavior
Build, a convolution neural network in Keras that predicts steering angles from images
Train and validate the model with a training and validation set
Test that the model successfully drives around track one without leaving the road
Summarize the results with a written report

[model]: ./images/model.png
[steering_histogram]: ./images/steering_histogram.png
[jittered_center_camera]: ./images/jittered_center_camera.png
[car_driving]: ./images/car_driving.png
[3cameras]: ./images/3cameras.png

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
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



### Model Architecture and Training
#### 1. Model Architecture
##### Input

The input is 66x200xC with C = 3 RGB color channels.

##### Architecture
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
### Output

**Layer Fully Connected** with 1 output value for the steering angle.

#### Visualisation
![Model][model]

#### 2. Overfitting
The model contains dropout layers in order to reduce overfitting
The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Training Data and process

Here are some visualisation of the training data:
![car_driving][car_driving]

![3cameras][3cameras]


Data was captured from the simulator. I drove conservatively around the track three times paying particular attention to the sharp right turn. I found connecting a PS3 controller allowed finer control then using the keyboard. At least once I waited till the last moment before taking the turn. This seems to have stopped the car ending up in the lake. Its also helped to overcome a symptom of the bias in the training data towards left turns. To further offset this risk, I validated the training using a test set I'd captured from the second track, which is a lot more windy.


The Adam Optimizer was used with a mean squared error loss. A number of hyper-parameters were passed on the command line. The command I used looks such for a batch size of 500, 10 epochs (dropped out early if loss wasn't improving), dropout at .25 with a training size of 50000 randomly augmented features with adjusted labels and 2000 random features & labels used for validation
The reason for choosing Adam optimizer is that it works well with sparse data, it adds bias correction and momentum. One can refer to the following article. http://sebastianruder.com/optimizing-gradient-descent/index.html#adam

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
I  randomly shuffled the data set and put 20% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
The simulator captures data into a csv log file which references left, centre and right captured images within a sub directory. Telemetry data for steering, throttle, brake and speed is also contained in the log. Only steering was used in this project.
Before being fed into the model, the images are cropped to 66x200 starting at height 60 with width centered -

As seen in the following histogram a significant proportion of the data is for driving straight and its lopsided to left turns (being a negative steering angle is left) when using data generated following my conservative driving laps.
![steering_histogram][steering_histogram]

The log file was preprocessed to remove contiguous rows with a history of >5 records, with a 0.0 steering angle. This was the only preprocessing done outside of the batch generators used in training (random rows are augmented/jittered for each batch at model training time).

Jittering was applied per [Vivek Yadav's post ](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0) to augment data. Images were randomly transformed in the x range by 100 pixels and in the y range by 10 pixels with 0.4 per xpixel adjusted against the steering angle. Brightness via a HSV (V channel) transform (.25 + a random number in range 0 to 1) was also performed.
![jittered_center_camera][jittered_center_camera]


During batch generation, to compensate for the left turning, 50% of images were flipped (including reversing steering angle) if the absolute steering angle was > .1.

Finally images are cropped per above before being batched.

The model was trained with 10 epochs

python model.py --batch_size=500 --training_log_path=./data --validation_log_path=./datat2 --epochs 10 \
--training_size 50000 --validation_size 2000 --dropout .25


#### 5. Solution design

I chose to use the [Nvidia's End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) model. I diverged by passing cropped camera images as RGB, and not YUV, with adjusting brightness and by using the steering angle as is.

I also experimented with [comma.ai](http://comma.ai/), [Steering angle prediction model](https://github.com/commaai/research/blob/master/train_steering_model.py). However, I didn't notice a better result with the same training data and decided to stick to NVIDIA model and improve my training data. 

The model represented here is what I used, it is an implementation of the nvidia model. It is developed in python using keras (High Level Tensorflow library) in model.py and returned from the `build_nvidia_model` method. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
Whenever I found that my  model had a low mean squared error on the training set but a high mean squared error on the validation set., this implied that the model was overfitting. 

To combat the overfitting, I generated better data. I augmented the training data by using mirror images of existing data. I also cropped the pictures to focus on the road.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


The model trained (which is saved), is used again in testing. The simulator feeds you the centre camera image, along with steering and throttle telemetry. In response you have to return the new steering angle and throttle values. I hard coded the throttle to .35. The image was cropped, the same as for training, then fed into the model for prediction giving the steering angle.

```python
steering_angle = float(model.predict(transformed_image_array, batch_size=1))
throttle = 0.35
```
References: All ideas are heavily taken from discussions on udacity Self driving car nanodegree cohort facebook community.
