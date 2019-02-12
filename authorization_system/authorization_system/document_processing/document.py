# coding=utf-8
"""
Implementation of Document class.
"""

import re
import time

import cv2
import numpy as np
import pytesseract

COLOR_FILLING = 170
CLEAN_WHITESPACES = dict(zip(" \n", "  "))
TYPE_MAP = {
    "id_card_new": "New ID card",
    "id_card_old": "Old ID card",
    "driving_license_new": "New driving license",
    "driving_license_old": "Old driving license",
    "student_card": "Student card",
    "unknown_document": "Unknown document",
}


class Document:
    def __init__(self):
        self.type = None
        self.type_auto = None
        self.image = None
        self.photo = None
        self.name_image = None
        self.box_name_image = None
        self.flipped_image = None
        self.name_detected_text = None
        self.name_lower_left_corner = None
        self.first_name_detected_text = None
        self.surname_detected_text = None
        self.elements_to_hide_bounding_boxes = None
        self.recognised = None
        self.fraud_try = None
        self.timestamp = str(int(time.time()))

    @property
    def full_name(self):
        return " ".join([self.first_name_detected_text, self.surname_detected_text])

    def set_image(self, image):
        if image is not None and len(image.shape) == 3:
            self.image = image.copy()
        else:
            self.image = None

    def _is_old_driving_license(self):
        img = self.image.copy()
        cut = img[280:300, 180:200, :]
        mean = cut.mean(axis=(0, 1))
        if mean[0] < 100 and mean[1] < 120 and mean[2] > 120:
            return True
        else:
            return False

    def _is_new_driving_license(self):
        img = self.image.copy()
        cut = img[170:190, 140:160, :]
        mean = cut.mean(axis=(0, 1))
        if mean[0] < 90 and mean[1] < 70 and mean[2] > 130:
            return True
        else:
            return False

    def _is_student_card(self):
        img = self.image.copy()
        cut = img[:30, :30, :]
        mean = cut.mean(axis=(0, 1))
        if mean[0] < 90 and mean[1] > 120 and mean[2] > 80:
            return True
        else:
            return False

    def _is_new_id_card(self):
        img = self.image.copy()
        cut = img[40:55, 55:70, :]
        mean = cut.mean(axis=(0, 1))
        if mean[0] > 150 and mean[1] < 70 and mean[2] < 90:
            return True
        else:
            return False

    def _is_old_id_card(self):
        img = self.image.copy()
        cut = img[35:50, 1600:1615, :]
        mean = cut.mean(axis=(0, 1))
        if mean[0] > 150 and mean[1] < 70 and mean[2] < 90:
            return True
        else:
            return False

    def _recognize_document_type(self):
        if self._is_old_driving_license():
            self.type = "driving_license_old"
        elif self._is_new_driving_license():
            self.type = "driving_license_new"
        elif self._is_student_card():
            self.type = "student_card"
        elif self._is_new_id_card():
            self.type = "id_card_new"
        elif self._is_old_id_card():
            self.type = "id_card_old"
        else:
            self.type = "unknown_document"

    def _set_type_dummy(self, type):
        if type == 1:
            self.type = "id_card_new"
        elif type == 2:
            self.type = "id_card_old"
        elif type == 3:
            self.type = "driving_license_new"
        elif type == 4:
            self.type = "driving_license_old"
        elif type == 5:
            self.type = "student_card"
        else:
            self.type = "unknown_document"

    def set_type(self, type=None, document_type_detector=None):
        if self.image is None:
            self.type = None
        else:
            for i in range(2):
                if document_type_detector is None:
                    self._recognize_document_type()
                else:
                    self.type = document_type_detector.detect(self.image.copy())
                if self.type != "unknown_document":
                    if self.flipped_image is None:
                        self.flipped_image = False
                    break
                self.image = cv2.flip(self.image, -1)
                if self.flipped_image is None:
                    self.flipped_image = True
                else:
                    self.flipped_image = not self.flipped_image
            self.type_auto = self.type
            if type is not None:
                self._set_type_dummy(type)

    def _find_hillside(self, vector, treshold, reverse=False, increase=True):
        if reverse:
            vector = vector[::-1]
        for i, e in enumerate(vector):
            if increase:
                if e > treshold:
                    break
            else:
                if e < treshold:
                    break
        return i

    def _sum_of_mins(self, img, axis, treshold_mins, width, colors=(0, 1, 2)):
        cut = cv2.medianBlur(img, 1)
        cut = cut[:, :, colors]
        mins = cut[:, :width].min(axis=2)
        min_ = np.percentile(mins, 2)
        mins = mins - min_
        max_ = np.percentile(mins, 98)
        mins = np.clip((mins * 255.0 / np.maximum(max_, 0.001)), 0, 255).astype("uint8")
        cut_min = (mins < treshold_mins) * 1.0
        m = cut_min.sum(axis=axis)
        return m

    def _trim_box(
        self,
        img,
        height_lower_bound,
        height_upper_bound,
        width_lower_bound,
        width_upper_bound,
        horizontal_trim_lower_margin,
        horizontal_trim_upper_margin,
        vertical_trim_lower_margin,
        hill_vert_treshold,
        hill_horiz_treshold,
        treshold_mins,
        width,
        colors,
    ):
        # choosing the part of the document with name with respect to the document's type
        img = img[
            height_lower_bound:height_upper_bound,
            width_lower_bound:width_upper_bound,
            :,
        ].copy()
        self.box_name_image = img.copy()

        # trimming chosen part with respect to it's properties
        som = self._sum_of_mins(img, 1, treshold_mins, width, colors)
        horizontal_trim = self._find_hillside(som, hill_horiz_treshold, increase=True)
        som = self._sum_of_mins(img, 0, treshold_mins, width, colors)
        vertical_trim = self._find_hillside(som, hill_vert_treshold, increase=True)
        horizontal_max = img.shape[0]
        img = img[
            max(0, (horizontal_trim + horizontal_trim_lower_margin)) : min(
                horizontal_trim + horizontal_trim_upper_margin, horizontal_max
            ),
            max(0, vertical_trim + vertical_trim_lower_margin) :,
            :,
        ]

        # calculating lower left corner of the trimmed part (for hiding sensitive data)
        lower_left_corner = (
            height_lower_bound
            + min(horizontal_trim + horizontal_trim_upper_margin, horizontal_max),
            width_lower_bound + max(0, vertical_trim + vertical_trim_lower_margin),
        )
        return img, lower_left_corner

    def _trim_name(self):
        img = self.image.copy()
        if self.type == "id_card_new":
            img, lower_left_corner = self._trim_box(
                img=img,
                height_lower_bound=300,
                height_upper_bound=800,
                width_lower_bound=650,
                width_upper_bound=1900,
                horizontal_trim_lower_margin=35,
                horizontal_trim_upper_margin=330,
                vertical_trim_lower_margin=-50,
                hill_vert_treshold=30,
                hill_horiz_treshold=50,
                treshold_mins=120,
                width=400,
                colors=[0, 1, 2],
            )
            img[130:190, :, :] = COLOR_FILLING
            return img, lower_left_corner
        elif self.type == "id_card_old":
            img, lower_left_corner = self._trim_box(
                img=img,
                height_lower_bound=290,
                height_upper_bound=650,
                width_lower_bound=650,
                width_upper_bound=1900,
                horizontal_trim_lower_margin=25,
                horizontal_trim_upper_margin=280,
                vertical_trim_lower_margin=-50,
                hill_vert_treshold=30,
                hill_horiz_treshold=40,
                treshold_mins=160,
                width=400,
                colors=[0],
            )
            img[110:155, :, :] = COLOR_FILLING
            return img, lower_left_corner
        elif self.type == "driving_license_old":
            img, lower_left_corner = self._trim_box(
                img=img,
                height_lower_bound=200,
                height_upper_bound=400,
                width_lower_bound=630,
                width_upper_bound=1900,
                horizontal_trim_lower_margin=-20,
                horizontal_trim_upper_margin=130,
                vertical_trim_lower_margin=50,
                hill_vert_treshold=20,
                hill_horiz_treshold=15,
                treshold_mins=150,
                width=200,
                colors=[0],
            )
            return img, lower_left_corner
        elif self.type == "driving_license_new":
            img, lower_left_corner = self._trim_box(
                img=img,
                height_lower_bound=170,
                height_upper_bound=400,
                width_lower_bound=600,
                width_upper_bound=1900,
                horizontal_trim_lower_margin=-30,
                horizontal_trim_upper_margin=140,
                vertical_trim_lower_margin=50,
                hill_vert_treshold=15,
                hill_horiz_treshold=15,
                treshold_mins=120,
                width=200,
                colors=[0],
            )
            return img, lower_left_corner
        elif self.type == "student_card":
            img = img[400:780, 550:1300, :].copy()
            self.box_name_image = img.copy()
            lower_left_corner = (780, 550)
            return img, lower_left_corner
        else:
            return None, None

    def set_document_bounding_boxes(self):
        height, width, _ = self.image.shape
        if self.type == "id_card_new":
            self.elements_to_hide_bounding_boxes = [
                [
                    self.name_lower_left_corner[1],
                    self.name_lower_left_corner[0],
                    width,
                    height,
                ]
            ]
        elif self.type == "id_card_old":
            self.elements_to_hide_bounding_boxes = [
                [
                    self.name_lower_left_corner[1],
                    self.name_lower_left_corner[0],
                    width,
                    height,
                ]
            ]
        elif self.type == "driving_license_new":
            self.elements_to_hide_bounding_boxes = [
                [
                    self.name_lower_left_corner[1],
                    self.name_lower_left_corner[0],
                    width,
                    height,
                ],
                [0, 1060, width, height],
                [0, 880, 165, 960],
                [140, 900, 235, 980],
                [210, 930, 395, 1010],
            ]
        elif self.type == "driving_license_old":
            self.elements_to_hide_bounding_boxes = [
                [
                    self.name_lower_left_corner[1],
                    self.name_lower_left_corner[0],
                    width,
                    height,
                ]
            ]
        elif self.type == "student_card":
            self.elements_to_hide_bounding_boxes = [[0, 780, 1300, height]]
        else:
            self.elements_to_hide_bounding_boxes = None

    def hide_elements(self):
        if self.elements_to_hide_bounding_boxes is not None:
            for element_to_hide in self.elements_to_hide_bounding_boxes:
                self.image[
                    element_to_hide[1] : element_to_hide[3],
                    element_to_hide[0] : element_to_hide[2],
                    :,
                ] = COLOR_FILLING

    def cut_personal_data(self):
        if self.image is not None:
            name_image, self.name_lower_left_corner = self._trim_name()
            if name_image is not None:
                name_image_with_border = np.zeros(
                    (name_image.shape[0] + 100, name_image.shape[1] + 100, 3),
                    dtype="uint8",
                )
                name_image_with_border[:] = COLOR_FILLING
                name_image_with_border[
                    50 : (50 + name_image.shape[0]), 50 : (50 + name_image.shape[1]), :
                ] = name_image
                self.name_image = name_image_with_border
            else:
                self.name_image = None
                self.name_lower_left_corner = None
        else:
            self.name_image = None
            self.name_lower_left_corner = None

    def _transform_text(self, image, treshold, blur):
        transformed = image.max(axis=2)
        transformed = cv2.medianBlur(transformed.astype("uint8"), blur)
        transformed_tresh = transformed > treshold
        transformed_tresh = cv2.medianBlur(
            transformed_tresh.astype("uint8") * 255, blur
        )
        transformed_tresh = transformed_tresh // 4 + (transformed / 4 * 3 // 1).astype(
            "uint8"
        )
        return transformed_tresh

    def _recognize_text(self, image):
        assert self.type is not None
        if self.type == "student_card":
            transformed = self._transform_text(image, 90, 5)
            text = pytesseract.image_to_string(transformed, lang="pol").strip().title()
        elif self.type == "driving_license_old":
            transformed = self._transform_text(image, 140, 5)
            text = pytesseract.image_to_string(transformed, lang="pol").strip().title()
        elif self.type == "id_card_new":
            transformed = self._transform_text(image, 100, 5)
            text = pytesseract.image_to_string(transformed, lang="pol").strip().title()
        else:
            transformed = self._transform_text(image, 110, 5)
            text = pytesseract.image_to_string(transformed, lang="pol").strip().title()
        if text in ["", " "]:
            text = None
        return text

    def _preprocess_text(self, name=None, title=True, dictionary=CLEAN_WHITESPACES):
        if name is not None:
            name = name.strip()
        else:
            name = self.name_detected_text.strip()
        if title:
            name = name.title()
        dictionary_name = "".join(
            map(lambda x: dictionary[x] if x in dictionary.keys() else x, name)
        )
        dictionary_name = re.sub(" +", " ", dictionary_name)
        return dictionary_name

    def _extract_full_name(self, dictionary=CLEAN_WHITESPACES):
        if (
            self.name_detected_text is None
            or len(re.sub("\n", " ", self.name_detected_text).split(" ")) == 1
        ):
            return None, None
        elif self.type == "student_card":
            text = self._preprocess_text(dictionary=dictionary)
            text = text.title().split(" ")
            return text[0], text[-1]
        elif self.type in (
            "id_card_new",
            "id_card_old",
            "driving_license_new",
            "driving_license_old",
        ):
            text = self._preprocess_text(dictionary=dictionary)
            text = text.title().split(" ")
            return text[1], text[0]
        elif self.type is None:
            text = self._preprocess_text(dictionary=dictionary)
            text = text.title().split(" ")
            return text[0], text[-1]
        else:
            return None, None

    def _recognize_photo(self, face_recognition=None):
        if self.image is None:
            document_face = None
        else:
            document_faces = face_recognition.identify(self.image, True)
            if len(document_faces) > 0:
                document_faces_size = [
                    (x[2] - x[0]) * (x[3] - x[1])
                    for x in [y.bounding_box for y in document_faces]
                ]
                document_face = document_faces[np.argmax(document_faces_size)].image
            else:
                document_face = None
        return document_face

    def recognize_person(self, face_recognition=None, dictionary=CLEAN_WHITESPACES):
        if self.name_image is not None:
            self.name_detected_text = self._recognize_text(self.name_image.copy())
            self.first_name_detected_text, self.surname_detected_text = self._extract_full_name(
                dictionary
            )
        else:
            self.name_detected_text = None
        if self.image is not None and face_recognition is not None:
            self.photo = self._recognize_photo(face_recognition)
        else:
            self.photo = None

    def process_document(
        self, image, type=None, document_type_detector=None, face_recognition=None
    ):
        self.set_image(image)
        self.set_type(type, document_type_detector)
        self.cut_personal_data()
        self.set_document_bounding_boxes()
        self.hide_elements()
        self.recognize_person(face_recognition)

    def print_data(self):
        fields = {
            "type": self.type,
            "first_name_detected_text": self.first_name_detected_text,
            "surname_detected_text": self.surname_detected_text,
            "recognised": self.recognised,
            "fraud_try": self.fraud_try,
        }
        for k, v in fields.items():
            print(k, v)
