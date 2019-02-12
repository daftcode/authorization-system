# coding=utf-8
"""
Document type detection based on pixels' means.
"""

import pickle

import cv2
import numpy as np


class DocumentTypeDetector:
    def __init__(self, stride_size=10, treshold=0.22):
        self.stride_size = stride_size
        self.document_means = None
        self.document_names = [
            "id_card_old",
            "id_card_new",
            "student_card",
            "driving_license_old",
            "driving_license_new",
        ]
        self.treshold = treshold

    def avg_square(self, img, size=5):
        img = img.copy()
        for i in range(0, img.shape[0], size):
            for j in range(0, img.shape[1], size):
                img[i : (i + size), j : (j + size), :] = img[
                    i : (i + size), j : (j + size), :
                ].mean(axis=(0, 1))
        return img

    def fit(self, files, n_aug=10, aug_range=5):
        self.document_means = []
        for doc_type in self.document_names:
            # for each document's type augment data based on the images
            images_tmp = []
            for file in files:
                if doc_type in file:
                    img = cv2.imread(file)[:, :, :3]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    assert img.shape == (1180, 1960, 3)
                    for __ in range(n_aug):
                        img_aug = np.roll(
                            img, np.random.randint(-aug_range, aug_range), axis=0
                        )
                        img_aug = np.roll(
                            img_aug, np.random.randint(-aug_range, aug_range), axis=1
                        )
                        images_tmp.append(img_aug)

            # calculating mean from augmented images
            m_img = np.stack(images_tmp).mean(axis=0)
            m_img = m_img.astype("uint8")
            self.document_means.append(m_img)
        return self

    def detect(self, img):
        assert img.shape == (1180, 1960, 3)
        # blurring image
        img_t = cv2.medianBlur(img, 3)
        img_t = img_t[50:300, 50:650, :]
        img_t = self.avg_square(img_t, self.stride_size)
        dist = np.ones(len(self.document_names))

        for i, img_doc in enumerate(self.document_means):
            # finding most similar document type
            assert img_doc.shape == (1180, 1960, 3)
            img_doc_t = cv2.medianBlur(img_doc, 3)
            img_doc_t = img_doc_t[50:300, 50:650, :]
            img_doc_t = self.avg_square(img_doc_t, self.stride_size)
            diff = np.abs(img_t.astype("float32") - img_doc_t.astype("float32")).sum(
                axis=2
            )
            diff = diff > 100
            dist[i] = diff.mean()

        # choosing document's type
        if dist.min() < self.treshold:
            return self.document_names[np.argmin(dist)]
        else:
            return "unknown_document"

    def save_document_means(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.document_means, f, pickle.HIGHEST_PROTOCOL)

    def load_document_means(self, filename):
        with open(filename, "rb") as f:
            self.document_means = pickle.load(f)
