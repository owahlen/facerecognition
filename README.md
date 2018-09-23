# facerecognition

Recognize people from an image of their face.

## Introduction
The project is based on this tutorial of Cole Murray:
[Building a Facial Recognition Pipeline with Deep Learning in Tensorflow](https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8).
It uses preprocessed images of the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset
to train an
[Inception Resnet V1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1)
Tensorflow based convolutional neuronal network.
The preprocessing step uses [dlib](http://dlib.net)’s face landmark predictor to crop the images to the relevant part of the face.

## Installation Requirements
* [python 3.6](https://apple.stackexchange.com/questions/329187/homebrew-rollback-from-python-3-7-to-python-3-6-5-x)
* [tensorflow](https://www.tensorflow.org/install/source)
* [opencv](https://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5)
* [dlib](https://www.learnopencv.com/install-dlib-on-macos)

A detailed script on how to install these components on a Linux machine can be found in the
[Dockerfile](https://github.com/ColeMurray/medium-facenet-tutorial/blob/master/Dockerfile) of the
[medium-facenet-tutorial](https://github.com/ColeMurray/medium-facenet-tutorial) Github tutorial.

## Project Execution
All execution steps are defined in the `Makefile` and executed using `make`.
Note that the command `make clean` can be used to remove all temporary files.

### Step 1: Download data files
Files that need to be downloaded are:
* the _Labeled Faces in the Wild_ data set
* the dlib _face landmark predictor_
* the pretrained weights of the Inception Resnet V1 model

To download and unpack all needed files into the `data` and `model` folders execute:
```
$ make download
```

### Step 2: Preprocess the image files
In order to train the tensorflow model the input images must be normalized.
First the largest face is identified in each LFW image.
Then the image is cropped and centered by the inner eyes and the bottom lip and scaled to 180x180 pixels.
The preprocessing step is achieved by executing the command:
```
$ make preprocess
```
The processed images are stored in the `output/intermediate` folder.

### Step 3: Train the Inception Resnet V1 Tensorflow model
To train the model execute the command:
```
$ make train
```
The training result is stored in the file `output/classifier.pkl`.

### Step 4: Evaluate the quality of the classifier
To evaluate the quality of the classifier execute the command:
```
$ make test
```
This will print the detection accuracy for each image.
The overall accuracy should be around 90%.
