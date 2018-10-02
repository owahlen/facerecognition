# facerecognition

Recognize people from an image of their face.

## Introduction
The project is heavily based on the
[facenet](https://github.com/davidsandberg/facenet)
project by David Sandberg.
It uses preprocessed images of the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset
to train an
[Inception ResNet V1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1)
Tensorflow based convolutional neuronal network.

## Installation Requirements
* [python 3.6](https://apple.stackexchange.com/questions/329187/homebrew-rollback-from-python-3-7-to-python-3-6-5-x)
* [tensorflow](https://www.tensorflow.org/install/source)

## Project Execution
All execution steps are defined in the `Makefile` and executed using `make`.
Note that the command `make clean` can be used to remove all temporary files.

### Step 1: Download data files
Files that need to be downloaded are:
* the _Labeled Faces in the Wild_ data set
* the pretrained weights of the Inception Resnet V1 model

To download and unpack all needed files into the `datasets` and `model` folders execute:
```
$ make download
```

### Step 2: Align the image files
As a proprocessing step the images of the faces must be normalized.
More specifically they must be cropped and centered by the inner eyes and the bottom lip and scaled to 160x160 pixels.
This is achieved with a [Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html).
The Python/Tensorflow code is contained in the _src/align_ directory.

Alignment is executing with the command:
```
$ make align
```
The processed images are stored in the `datasets/lfw/lfw_mtcnnpy_160` folder.

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
