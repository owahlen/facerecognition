#!/usr/bin/env python3

import argparse
import json
import sys

import cv2
import numpy as np
import zmq

import face_detector as detector


def main(args):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:%d' % (args.port))

    face_detector = detector.FaceDetector(args.detect_multiple_faces)

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
