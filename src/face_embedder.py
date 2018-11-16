import math

import numpy as np
import tensorflow as tf


class FaceEmbedder():

    def __init__(self, model):
        self._batch_size = 90
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
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self._batch_size))
        embeddings = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * self._batch_size
            end_index = min((i + 1) * self._batch_size, nrof_images)
            feed_dict = {images_placeholder: face_images, phase_train_placeholder: False}
            embeddings[start_index:end_index, :] = session.run(embeddings_tensor, feed_dict=feed_dict)

        return embeddings
