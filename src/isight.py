#!/usr/bin/env python

import sys
import argparse
import cv2
from os.path import realpath, normpath

class FaceDetector():
    def __init__(self):
        cascade_dir = normpath(realpath(cv2.__file__) + '/../data')
        cascade_file = cascade_dir + '/haarcascade_frontalface_alt.xml'
        self.classifier = cv2.CascadeClassifier(cascade_file)

    def detect(self, image):
        image_copy = image.copy()
        gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        return self.classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)


def main(args):

    PREVIEW_WINDOW = 'preview'
    ESCAPE_KEY = 27

    cv2.namedWindow(PREVIEW_WINDOW)
    vc = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    doCapture = True

    while doCapture:
        rval, frame = vc.read()
        if frame is not None:
            faces = face_detector.detect(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow(PREVIEW_WINDOW, frame)
        if cv2.waitKey(1) == ESCAPE_KEY or cv2.getWindowProperty(PREVIEW_WINDOW,cv2.WND_PROP_VISIBLE) < 1:
            doCapture = False

    cv2.destroyWindow(PREVIEW_WINDOW)

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Take images from isight camera')
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
