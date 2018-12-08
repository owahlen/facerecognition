import math

import numpy as np
import tensorflow as tf


class FaceEmbedder():

    def __init__(self, model):
        with tf.gfile.FastGFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def compute(self, session, face_images):
        images_placeholder = session.graph.get_tensor_by_name("input:0")
        embeddings_tensor = session.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = session.graph.get_tensor_by_name("phase_train:0")
        embedding_size = embeddings_tensor.get_shape()[1]

        nrof_images = len(face_images)
        if nrof_images == 0:
            return []

        images = []
        for i in range(nrof_images):
            image = tf.image.resize_image_with_crop_or_pad(face_images[i], 160, 160)
            image = tf.image.per_image_standardization(image)
            image = image.eval()
            images.append(image)


        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        embeddings = session.run(embeddings_tensor, feed_dict=feed_dict)

        return embeddings
