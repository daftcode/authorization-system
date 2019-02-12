# coding=utf-8
"""
Utils for face recognition.
"""

import cv2
from authorization_system.face_recognition import face


def set_face_model(args, verbose=True):
    if args.facenet_model_checkpoint_dir != "./model/facenet":
        if verbose:
            print(
                "Default facenet model checkpoint directory changed to: "
                + args.facenet_model_checkpoint_dir
            )
        face.FACENET_MODEL_CHECKPOINT = args.facenet_model_checkpoint_dir


def add_overlays(frame, faces, scale=1, realsense=False):
    frame_width = frame.shape[1]
    if realsense:
        scale_text = scale + 1
    else:
        scale_text = scale

    if faces is not None:
        # showing bounding boxes
        nr_faces = len(faces)
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            if nr_faces > 1:
                color = (35, 35, 220)
            else:
                color = (30, 160, 30)
            cv2.rectangle(
                frame,
                (frame_width - scale * face_bb[2], scale * face_bb[1]),
                (frame_width - scale * face_bb[0], scale * face_bb[3]),
                color,
                2,
            )

    # showing information if two people are on the camera
    if nr_faces > 1:
        cv2.putText(
            frame,
            "At least two people on the camera",
            (scale_text * 45, scale_text * 60),
            cv2.FONT_HERSHEY_COMPLEX,
            scale_text * 0.9,
            (35, 35, 220),
            thickness=2,
        )
