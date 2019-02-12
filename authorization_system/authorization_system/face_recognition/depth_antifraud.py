# coding=utf-8
"""
Antifraud based on depth images.
Intel RealSense Depth Camera D435 was tested.
"""

import numpy as np


class DepthAntifraud:
    def __init__(self, args):
        self.args = args

    def fill_zeros(self, img, border=1, times=3):
        # replacing zeros in the depth image with means of nearby points
        img = img.astype("float")
        filling = []
        for _ in range(times):
            for i in range(border, img.shape[0] - border):
                for j in range(border, img.shape[1] - border):
                    if img[i, j] == 0:
                        slice_ = img[
                            (i - border) : (i + border), (j - border) : (j + border)
                        ]
                        value = np.mean(slice_[slice_ > 0])
                        if np.isnan(value):
                            value = 0
                        filling.append((i, j, value))
            for i, j, value in filling:
                img[i, j] = value
        return img

    def find_eyes(
        self,
        face_img,
        ignore_zeros=False,
        start_r=10,
        area_r=2,
        repeat_times=5,
        alpha_flip=0,
        move_nose_y=0,
    ):

        # vertical rotating of "face surface"
        if alpha_flip != 0:
            flip_matrix_column = (
                np.linspace(0, face_img.shape[0] * alpha_flip, face_img.shape[0])
                - face_img.shape[0] * alpha_flip / 2
            )
            flip_matrix = np.zeros(face_img.shape)
            for i in range(face_img.shape[1]):
                flip_matrix[:, i] = -flip_matrix_column
            face_img = face_img.copy() + flip_matrix

        # finding potential nose (the closest point to the camera)
        if ignore_zeros:
            face_img_min = np.min(face_img[np.where(face_img != 0)])
            minis = np.where(face_img == face_img_min)
        else:
            minis = np.where(face_img == face_img.min())

        # creating start points for finding eyes
        x, y = int(np.median(minis[1])), int(np.median(minis[0])) + move_nose_y
        y = np.clip(y, 0, face_img.shape[0])
        eyes_x = []
        eyes_y = []
        starts = [
            (x, y),
            (max(0, x - start_r), y),
            (max(0, x - 2 * start_r), max(0, y - 2 * start_r)),
            (x, max(0, y - start_r)),
            (min(face_img.shape[1], x + 2 * start_r), max(0, y - 2 * start_r)),
            (min(face_img.shape[1], x + start_r), y),
        ]

        # finding points (potential eyes) which are the farthest to the camera
        # with similar idea to gradient ascent
        for (x, y) in starts:
            for _ in range(repeat_times):
                max_x = -1
                max_y = -1
                max_value = -1
                while not (max_x == x and max_y == y):
                    area = face_img[
                        max(0, y - area_r) : min(y + area_r, face_img.shape[0]),
                        max(0, x - area_r) : min(x + area_r, face_img.shape[1]),
                    ]
                    area_max_value = area.max()
                    max_x = x
                    max_y = y
                    if area_max_value > max_value:
                        max_value = area_max_value
                        where_max = np.where(area == area_max_value)
                        choose_to_go = np.random.randint(len(where_max[0]))
                        x -= area_r - where_max[0][choose_to_go]
                        y -= area_r - where_max[1][choose_to_go]
                        x = x.clip(0, face_img.shape[1])
                        y = y.clip(0, face_img.shape[0])
                eyes_x.append(max_x)
                eyes_y.append(max_y)
        return eyes_x, eyes_y

    def check_depth_faces(
        self,
        depth_faces,
        ignore_zeros=True,
        fill_zeros_times=0,
        height_r_cut=0.1,
        width_r_cut=0.25,
        start_r=10,
        area_r=2,
        repeat_times=5,
        height_r_middle_det=0.25,
        width_r_middle_det=0.1,
        alpha_flip=0,
        move_nose_y=0,
    ):
        if depth_faces is None or len(depth_faces) == 0:
            return False, None
        real_faces = []
        for face in depth_faces:
            try:
                # extracting middle of the face
                height, width = face.shape
                middle_face = face.copy()[
                    int(height_r_cut * height) : int((1 - height_r_cut) * height),
                    int(width_r_cut * width) : int((1 - width_r_cut) * width),
                ]
                if not ignore_zeros:
                    middle_face = self.fill_zeros(middle_face, times=fill_zeros_times)

                # searching for the eyes
                eyes_xs, eyes_ys = self.find_eyes(
                    middle_face,
                    ignore_zeros=ignore_zeros,
                    start_r=start_r,
                    area_r=area_r,
                    repeat_times=repeat_times,
                    alpha_flip=alpha_flip,
                    move_nose_y=move_nose_y,
                )

                # checking how many times the eyes were found in the final area of the face
                middle_height, middle_width = middle_face.shape
                final_height_lower = height_r_middle_det * middle_height
                final_height_upper = (1 - height_r_middle_det) * middle_height
                final_width_lower = width_r_middle_det * middle_width
                final_width_upper = (1 - width_r_middle_det) * middle_width
                in_final_range = []
                for i in range(len(eyes_xs)):
                    if (final_height_lower < eyes_ys[i] < final_height_upper) and (
                        final_width_lower < eyes_xs[i] < final_width_upper
                    ):
                        in_final_range.append(1)
                    else:
                        in_final_range.append(0)

                # checking if real face
                real_face = (
                    np.sum(in_final_range) / len(in_final_range)
                ) > self.args.real_face_treshold
            except ValueError:
                real_face = False
            real_faces.append(real_face)

        # checking if real person
        real_person = (
            np.sum(real_faces) / len(real_faces)
        ) > self.args.real_person_treshold
        return real_person, real_faces
