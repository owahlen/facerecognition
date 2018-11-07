#!/usr/bin/env python3

import sys
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import cv2
import zmq
import json


class FaceDetector():
    def __init__(self, detect_multiple_faces):
        self.detect_multiple_faces = detect_multiple_faces

        with tf.Graph().as_default():
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess, None)

        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor

    def detect(self, image):
        img = image.copy()
        if img.ndim < 2:
            raise ValueError('image has ndim<2')
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:, :, 0:3]

        bounding_boxes, _ = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        nrof_faces = bounding_boxes.shape[0]
        det_arr = []
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces > 1:
                if self.detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                         (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    index = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                    det_arr.append(det[index, :].astype(int))
            else:
                det_arr.append(np.squeeze(det.astype(int)))
        return np.array(det_arr)


def main(args):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:%d' % (args.port))

    face_detector = FaceDetector(args.detect_multiple_faces)

    while True:
        img = socket.recv()
        npimg = np.fromstring(img, dtype=np.uint8)
        frame = cv2.imdecode(npimg, 1)
        if frame is not None:
            faces = face_detector.detect(frame)
            response = json.dumps(faces.tolist())
            socket.send_string(response)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Take images from isight camera')

    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    parser.add_argument('--port', type=int,
                        help='Receiving port', default=5555)

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
