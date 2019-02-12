# coding=utf-8
"""
Face Detection and Recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

import numpy as np
import tensorflow as tf
from facenet.src import facenet
from facenet.src.align import detect_face
from scipy import misc

FACENET_MODEL_CHECKPOINT = "./models/facenet/"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Recognition:
    def __init__(
        self, face_crop_size=160, face_crop_margin=44, gpu_memory_fraction=0.3
    ):
        self.detect = Detection(
            face_crop_size=face_crop_size,
            face_crop_margin=face_crop_margin,
            gpu_memory_fraction=gpu_memory_fraction,
        )
        self.encoder = Encoder()

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)
        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image, only_detect=False, frame_downscale=1):
        faces = self.detect.find_faces(image, frame_downscale)
        if not only_detect:
            for i, face in enumerate(faces):
                face.embedding = self.encoder.generate_embedding(face)
        return faces


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(FACENET_MODEL_CHECKPOINT)

    def generate_embedding(self, face):
        images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
        embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")
        prewhiten_face = facenet.prewhiten(face.image)
        feed_dict = {
            images_placeholder: [prewhiten_face],
            phase_train_placeholder: False,
        }
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(
        self, face_crop_size=160, face_crop_margin=44, gpu_memory_fraction=0.3
    ):
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.gpu_memory_fraction = gpu_memory_fraction
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=self.gpu_memory_fraction
            )
            sess = tf.Session(
                config=tf.ConfigProto(
                    gpu_options=gpu_options, log_device_placement=False
                )
            )
            with sess.as_default():
                return detect_face.create_mtcnn(sess, None)

    def find_faces(self, image, frame_downscale):
        faces = []
        # finding faces with frame downscale
        bounding_boxes, _ = detect_face.detect_face(
            image[::frame_downscale, ::frame_downscale, :],
            self.minsize,
            self.pnet,
            self.rnet,
            self.onet,
            self.threshold,
            self.factor,
        )

        for bb in bounding_boxes:
            # updating bounding boxes wrt downscaling
            bb = bb * frame_downscale
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)
            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(
                bb[2] + self.face_crop_margin / 2, img_size[1]
            )
            face.bounding_box[3] = np.minimum(
                bb[3] + self.face_crop_margin / 2, img_size[0]
            )

            # cropping and resizing image
            cropped = image[
                face.bounding_box[1] : face.bounding_box[3],
                face.bounding_box[0] : face.bounding_box[2],
                :,
            ]
            face.image = misc.imresize(
                cropped, (self.face_crop_size, self.face_crop_size), interp="bilinear"
            )
            faces.append(face)
        return faces
