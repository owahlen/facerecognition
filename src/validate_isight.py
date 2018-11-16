import argparse
import sys

import cv2
import numpy as np
import tensorflow as tf

import face_detector as detector
import face_embedder as embedder
import face_extractor as extractor
import facenet


def main(args):
    PREVIEW_WINDOW = 'preview'
    QUIT_KEY = 27  # Escape
    ANCHOR_KEY = ord(' ')
    WIDTH = 640
    THRESHOLD = 0.39

    face_detector = detector.FaceDetector(args.detect_multiple_faces)
    face_extractor = extractor.FaceExtractor(160, 44)
    face_embedder = embedder.FaceEmbedder(args.model)

    cv2.namedWindow(PREVIEW_WINDOW)
    vc = cv2.VideoCapture(0)

    doCapture = True
    anchor_embedding = None
    with tf.Session() as sess:
        while doCapture:
            rval, frame = vc.read()
            if frame is None:
                continue

            height = int(frame.shape[0] * WIDTH / frame.shape[1])
            frame = cv2.resize(frame, (WIDTH, height))  # resize the frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert frame to RGB space

            # analysis
            boxes = face_detector.detect(image)
            faces = face_extractor.extract(image, boxes)
            embeddings = face_embedder.compute(sess, faces)

            # rendering of results
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if isinstance(anchor_embedding, np.ndarray):
                    distance = facenet.distance([embeddings[i]], [anchor_embedding])
                    if distance <= THRESHOLD:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    frame = cv2.putText(frame,
                                        str("{:.6f}".format(distance[0])),
                                        (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        color, 1, cv2.LINE_AA)
                else:
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.imshow(PREVIEW_WINDOW, frame)

            # key handling
            key = cv2.waitKey(1)
            if key == QUIT_KEY or cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                doCapture = False
            if key == ANCHOR_KEY and len(embeddings) > 0:
                anchor_embedding = embeddings[0]

    cv2.destroyWindow(PREVIEW_WINDOW)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Learn face from isight camera and recognize it later')
    parser.add_argument('--model', type=str,
                        help='Path to a model protobuf (.pb) file',
                        default='model/20181112-063708/20181112-063708.pb')
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.',
                        default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
