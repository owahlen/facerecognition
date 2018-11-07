#!/usr/bin/env python

import argparse
import json
import sys
import cv2
import zmq


def main(args):
    PREVIEW_WINDOW = 'preview'
    ESCAPE_KEY = 27
    WIDTH = 640

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://%s:%d' % (args.host, args.port))

    cv2.namedWindow(PREVIEW_WINDOW)
    vc = cv2.VideoCapture(0)

    doCapture = True
    while doCapture:
        rval, frame = vc.read()
        if frame is not None:
            height = int(frame.shape[0] * WIDTH / frame.shape[1])
            frame = cv2.resize(frame, (WIDTH, height))  # resize the frame
            encoded, buffer = cv2.imencode('.jpg', frame)
            socket.send(buffer)
            json_result = socket.recv_string()
            faces = json.loads(json_result)
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
    parser.add_argument('--host',
                        help='Destination host of the stream', default='localhost')
    parser.add_argument('--port', type=int,
                        help='Destination port of the stream', default=5555)

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
