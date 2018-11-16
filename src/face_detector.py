import numpy as np
import tensorflow as tf

import align.detect_face


class FaceDetector():

    def __init__(self, detect_multiple_faces):
        self.detect_multiple_faces = detect_multiple_faces

        with tf.Graph().as_default():
            config = tf.ConfigProto(log_device_placement=False)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            with sess.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess, None)

        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor

    def detect(self, image):
        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)

        nrof_faces = bounding_boxes.shape[0]
        boxes = []

        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            img_size = np.asarray(image.shape)[0:2] # [h, w]
            if nrof_faces > 1:
                if self.detect_multiple_faces:
                    for i in range(nrof_faces):
                        boxes.append(np.squeeze(det.astype(int)[i]))
                else:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]) # (x2-x1)*(y2-y1)
                    img_center = img_size / 2 # [h/2, w/2]
                    offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                         (det[:, 1] + det[:, 3]) / 2 - img_center[0]]) # [(x2+x1)/2-w/2, (y2-y1)/2*h/2]
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0) # sum( dx^2 + dy^2 )
                    index = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                    # append boxes in order of relevance: box_size - 2*distance_from_center
                    boxes.append(det[index, :].astype(int))
            else:
                boxes.append(np.squeeze(det.astype(int)))
        return np.array(boxes)
