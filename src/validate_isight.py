import tensorflow as tf
import numpy as np
import argparse
import facenet
import math
from scipy import misc
import align.detect_face
import pickle
import cv2


def classify(args, pnet, rnet, onet):
    _pnet = pnet
    _rnet = rnet
    _onet = onet
    BATCH_SIZE = 90
    IMAGE_SIZE = 160
    MARGIN = 44

    with tf.gfile.FastGFile(args.model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        images_placeholder = sess.graph.get_tensor_by_name("input:0")
        embeddings = sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        print('Calculating features for images')
        img = cv2.imread(args.image)
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
        nrof_images = len(images)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / BATCH_SIZE))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * BATCH_SIZE
            end_index = min((i + 1) * BATCH_SIZE, nrof_images)
            images, boxes = align_data(images, IMAGE_SIZE, MARGIN, _pnet, _rnet, _onet)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

        # at this point emb_array[0] contains the embedding of the image
        # boxes[0] represent the bounding box in the image

        classifier_filename_exp = args.classifier

        print('Testing classifier')
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        print('Loaded classifier model from file "%s"' % classifier_filename_exp)

        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
            img = cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])),
                                (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0),
                                2)
            img = cv2.putText(img, (class_names[best_class_indices[i]] + " - " + str(
                "{:.2f}%".format(best_class_probabilities[i] * 100))),
                              (int(boxes[i][0]), int(boxes[i][1] - 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('Picture classifier', img)
            res = cv2.waitKey(0)
            while res != 27:
                res = cv2.waitKey(0)

            exit(0)


def align_data(image_list, image_size, margin, pnet, rnet, onet):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_list = []
    boxes = []

    for x in range(len(image_list)):
        img_size = np.asarray(image_list[x].shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(image_list[x], minsize, pnet, rnet, onet, threshold, factor)
        nrof_samples = len(bounding_boxes)
        if nrof_samples > 0:
            for i in range(nrof_samples):
                if bounding_boxes[i][4] > 0.95:
                    det = np.squeeze(bounding_boxes[i, 0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = image_list[x][bb[1]:bb[3], bb[0]:bb[2], :]
                    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    prewhitened = facenet.prewhiten(aligned)
                    img_list.append(prewhitened)
                    boxes.append(bounding_boxes[i])

    if len(img_list) > 0:
        images = np.stack(img_list)
        return images, boxes

    return None, None


def create_network_face_detection():
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Path to a model protobuf (.pb) file')
    parser.add_argument('classifier',
                        help='Path to the classifier model file name as a pickle (.pkl) file.')
    parser.add_argument('image',
                        help='Path to the image file to classify.')

    args = parser.parse_args()

    pnet, rnet, onet = create_network_face_detection()

    classify(args, pnet, rnet, onet)