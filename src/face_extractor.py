import numpy as np
from scipy import misc


class FaceExtractor():

    def __init__(self, resize_to, margin):
        self.resize_to = resize_to
        self.margin = margin

    def extract(self, image, boxes):
        extracted_images = []

        image_size = np.asarray(image.shape)[0:2]
        for i, box in enumerate(boxes):
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(box[0] - self.margin / 2, 0)
            bb[1] = np.maximum(box[1] - self.margin / 2, 0)
            bb[2] = np.minimum(box[2] + self.margin / 2, image_size[1])
            bb[3] = np.minimum(box[3] + self.margin / 2, image_size[0])
            cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = misc.imresize(cropped, (self.resize_to, self.resize_to), interp='bilinear')
            extracted_images.append(scaled)

        return np.array(extracted_images)
