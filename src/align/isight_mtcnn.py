#!/usr/bin/env python

import sys
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import cv2


class FaceDetector():
    def __init__(self, gpu_memory_fraction, detect_multiple_faces):
        self.gpu_memory_fraction = gpu_memory_fraction
        self.detect_multiple_faces = detect_multiple_faces

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
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
        return det_arr


def main(args):
    PREVIEW_WINDOW = 'preview'
    ESCAPE_KEY = 27

    cv2.namedWindow(PREVIEW_WINDOW)
    vc = cv2.VideoCapture(0)
    face_detector = FaceDetector(args.gpu_memory_fraction, args.detect_multiple_faces)
    doCapture = True

    while doCapture:
        rval, frame = vc.read()
        if frame is not None:
            faces = face_detector.detect(frame)
            for (x1, y1, x2, y2) in faces:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(PREVIEW_WINDOW, frame)
        if cv2.waitKey(1) == ESCAPE_KEY or cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            doCapture = False

    cv2.destroyWindow(PREVIEW_WINDOW)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Take images from isight camera')

    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
