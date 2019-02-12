# coding=utf-8
"""
Face recording and comparision with scanned document.
"""

import glob
import os
import sys
import time
from termios import tcflush, TCIFLUSH

import cv2
import imageio
import numpy as np
import pyrealsense2 as rs
from authorization_system.document_processing.document_utils import BColors
from authorization_system.face_recognition import face, face_utils


class FaceRecorder:
    def __init__(self, args, face_recognition=None):
        self.args = args
        if face_recognition is None:
            face_utils.set_face_model(args, verbose=False)
            face_recognition = face.Recognition(
                face_crop_size=args.image_size,
                face_crop_margin=args.margin,
                gpu_memory_fraction=args.gpu_memory_fraction,
            )
        self.face_recognition = face_recognition

    def record_face(
        self,
        recording_time=10,
        scale=2,
        width=640,
        height=480,
        frame_downscale=2,
        frame_interval=3,
        realsense=False,
        window_recording_title="Face Recording",
        test_person_faces=None,
        verbose=True,
    ):
        if verbose:
            print("\nFace recording will last %d seconds." % recording_time)
        tcflush(sys.stdin, TCIFLUSH)
        input(
            (
                "\n%s%sMake sure you are alone on the camera. "
                + "To start recording press 'enter'.%s"
            )
            % (BColors.BOLD, BColors.WARNING, BColors.ENDC)
        )

        # choosing camera or test images
        if realsense and test_person_faces is None:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
            align = rs.align(rs.stream.color)
            pipeline.start(config)
        elif not realsense and test_person_faces is None:
            video_capture = cv2.VideoCapture(0)
        detected_faces = []
        detected_depth_faces = []
        frame_count = 0
        if test_person_faces is not None:
            nr_test_person_faces = len(test_person_faces)
            counter_test_person_faces = 0
        else:
            nr_test_person_faces = -1
            counter_test_person_faces = 0
        recording_start_time = time.time() + 1

        while True:
            # capturing frames from camera or test images
            if realsense and test_person_faces is None:
                frames_pipe = align.process(pipeline.wait_for_frames())
                frame = frames_pipe.get_color_frame()
                depth_frame = frames_pipe.get_depth_frame()
                if not frame or not depth_frame:
                    continue
                frame = np.asanyarray(frame.get_data())
                depth_frame = np.asanyarray(depth_frame.get_data())
            elif not realsense and test_person_faces is None:
                ret, frame = video_capture.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = test_person_faces[counter_test_person_faces]

            # detecting faces
            if (frame_count % frame_interval) == 0 or test_person_faces is not None:
                faces = self.face_recognition.identify(
                    frame, only_detect=True, frame_downscale=frame_downscale
                )
                time_from_start = time.time() - recording_start_time
                if len(faces) == 1 and (
                    time_from_start > 1 or test_person_faces is not None
                ):
                    detected_faces.append(faces[0].image)
                    if realsense and test_person_faces is None:
                        bb = faces[0].bounding_box
                        depth_face = depth_frame[bb[1] : bb[3], bb[0] : bb[2]]
                        detected_depth_faces.append(depth_face.astype("uint8"))

            # showing resized images with bounding boxes
            frame_count += 1
            if test_person_faces is not None:
                counter_test_person_faces += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)
            if not realsense:
                frame = cv2.resize(frame, (scale * width, scale * height))
            face_utils.add_overlays(frame, faces, scale=scale, realsense=realsense)
            cv2.imshow(window_recording_title, frame)
            if test_person_faces is not None:
                time.sleep(0.25)

            # stop recording condition
            if (
                (cv2.waitKey(1) & 0xFF == ord("q"))
                or (time_from_start > recording_time)
                or (
                    cv2.getWindowProperty(window_recording_title, cv2.WND_PROP_VISIBLE)
                    < 1
                )
                or counter_test_person_faces == nr_test_person_faces
            ):
                cv2.destroyAllWindows()
                break

        if verbose:
            print("\n%d images was recorded." % len(detected_faces))
        if realsense and test_person_faces is None:
            pipeline.stop()
        elif not realsense and test_person_faces is None:
            video_capture.release()
        return detected_faces, detected_depth_faces

    def check_if_correct_recording(
        self, detected_faces, min_nr_recorded_photos_treshold=40
    ):
        if (
            detected_faces is None
            or len(detected_faces) < min_nr_recorded_photos_treshold
        ):
            print(
                "\n%s%sToo few images was recorded. Try again.%s"
                % (BColors.BOLD, BColors.FAIL, BColors.ENDC)
            )
            return False
        else:
            return True

    def check_if_recognised(
        self,
        document_info,
        detected_faces,
        min_distance,
        min_nr_similar_photos_treshold,
        verbose=True,
    ):
        if verbose:
            print("\nYour document will be compared with recorded images right now.")
        if document_info.photo is None:
            if verbose:
                print("\nPhoto on the document wasn't detected.")
            document_info.recognised = False
            return document_info, 0
        elif detected_faces is None or len(detected_faces) == 0:
            if verbose:
                print("\nThere are no recorded images.")
            document_info.recognised = False
            return document_info, 0

        # generating color and gray embedding of the photo from the document
        image = document_info.photo.copy()
        document_color_face = face.Face()
        document_color_face.image = image
        document_color_face_embedding = self.face_recognition.encoder.generate_embedding(
            document_color_face
        )
        image = document_info.photo.copy()
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        for i in range(3):
            image[:, :, i] = gray_image
        document_gray_face = face.Face()
        document_gray_face.image = image.copy()
        document_gray_face_embedding = self.face_recognition.encoder.generate_embedding(
            document_gray_face
        )

        # creating color and gray embeddings of the faces
        detected_color_faces_embeddings = []
        detected_gray_faces_embeddings = []
        for img in detected_faces:
            image = img.copy()
            detected_color_face = face.Face()
            detected_color_face.image = image.copy()
            color_image_embedding = self.face_recognition.encoder.generate_embedding(
                detected_color_face
            )
            detected_color_faces_embeddings.append(color_image_embedding)
            image = img.copy()
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            for i in range(3):
                image[:, :, i] = gray_image
            detected_gray_face = face.Face()
            detected_gray_face.image = image.copy()
            gray_image_embedding = self.face_recognition.encoder.generate_embedding(
                detected_gray_face
            )
            detected_gray_faces_embeddings.append(gray_image_embedding)

        # computing (color images): distances between photo and faces;
        #                           number of similar pairs
        detected_color_faces_embeddings = np.array(detected_color_faces_embeddings)
        color_distances = np.array(
            [
                np.linalg.norm(x)
                for x in (
                    detected_color_faces_embeddings - document_color_face_embedding
                )
            ]
        )
        nr_similar_color_photos = np.sum(color_distances <= min_distance)

        # computing (gray images): distances between photo and faces;
        #                          number of similar pairs
        detected_gray_faces_embeddings = np.array(detected_gray_faces_embeddings)
        gray_distances = np.array(
            [
                np.linalg.norm(x)
                for x in (detected_gray_faces_embeddings - document_gray_face_embedding)
            ]
        )
        nr_similar_gray_photos = np.sum(gray_distances <= min_distance)

        # checking if the same person
        nr_similar_photos = max(nr_similar_color_photos, nr_similar_gray_photos)
        if nr_similar_photos < min_nr_similar_photos_treshold:
            if verbose:
                print("\nYou don't look like person on the document.")
            document_info.recognised = False
        else:
            if verbose:
                print(
                    (
                        "\nYou look like person on the document. "
                        + "You was recognised on %d images."
                    )
                    % nr_similar_photos
                )
            document_info.recognised = True
        return document_info, nr_similar_photos

    def save_recording(self, faces, depth_faces, document_info):
        if document_info is None:
            return None

        # creating paths or removing old files from these paths
        dir = self.args.data_save_dir
        person_name = document_info.timestamp + "_" + document_info.type
        person_name = person_name.replace(" ", "_")
        if os.path.exists(os.path.join(dir, "person_faces", person_name)):
            files = glob.glob(os.path.join(dir, "person_faces", person_name, "*"))
            for f in files:
                os.remove(f)
        else:
            os.makedirs(os.path.join(dir, "person_faces", person_name))
            if os.path.exists(os.path.join(dir, "person_depth_faces", person_name)):
                files = glob.glob(
                    os.path.join(dir, "person_depth_faces", person_name, "*")
                )
                for f in files:
                    os.remove(f)
            else:
                os.makedirs(os.path.join(dir, "person_depth_faces", person_name))

        # saving files: person's faces; depth person's faces
        if faces is not None:
            for i, face in enumerate(faces):
                imageio.imsave(
                    os.path.join(
                        dir,
                        "person_faces",
                        person_name,
                        person_name + "_" + str(i + 1) + ".jpg",
                    ),
                    face,
                )
        if depth_faces is not None:
            for i, depth_face in enumerate(depth_faces):
                imageio.imsave(
                    os.path.join(
                        dir,
                        "person_depth_faces",
                        person_name,
                        person_name + "_" + str(i + 1) + ".png",
                    ),
                    depth_face,
                )
