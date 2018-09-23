# facerecognition

Recognize people from an image of their face.

## Introduction
The project is based on this tutorial of Cole Murray:
[Building a Facial Recognition Pipeline with Deep Learning in Tensorflow](https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8).
It uses preprocessed images of the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset
to train a Tensorflow based convolutional neuronal network.
The preprocessing step uses [dlib](http://dlib.net)’s face landmark predictor to crop the images to the relevant part of the face.

## Installation Requirements
* [python 3.6](https://apple.stackexchange.com/questions/329187/homebrew-rollback-from-python-3-7-to-python-3-6-5-x)
* [tensorflow](https://www.tensorflow.org/install/source)
* [opencv](https://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5)
* [dlib](https://www.learnopencv.com/install-dlib-on-macos)

A detailed script on how to install these components on a Linux machine can be found in the `Dockerfile` of the
[medium-facenet-tutorial](https://github.com/ColeMurray/medium-facenet-tutorial) Github tutorial.

## Step 1: Download of the LFW data set and the dlib face landmark predictor
The `Makefile` contains the `curl`, `tar`, and `bzip2` commands
to download and unpack all needed files into the `data` folder:
```
$ make
```
Note that `make clean`can be used to remove all temporary files.

## Step 2: Preprocess the image files
In order to train the tensorflow model the images must be cropped and scaled to show only a standardized part of the face.
This preprocessing step is achieved by executing the command:
```
$ python preprocess.py
```
Note that the cropped images are stored in the `output` folder.
