# coding=utf-8
"""
Utils for document processing.
"""

import argparse
import glob
import os
import pickle

import cv2


class BColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def load_document_info_data(dir):
    documents_info_data = []
    documents_info_names = []
    files = glob.glob(os.path.join(dir, "*.pkl"))
    files = sorted(files)
    for file in files:
        with open(file, "rb") as f:
            data = pickle.load(f)
        documents_info_data.append(data)
        data_name = data.timestamp + "_" + data.type
        data_name = data_name.replace(" ", "_")
        documents_info_names.append(data_name)
    return documents_info_names, documents_info_data


def load_test_document_image(file_path):
    test_document_image = cv2.imread(file_path)
    test_document_image = cv2.cvtColor(test_document_image, cv2.COLOR_BGR2RGB)
    return test_document_image


def load_test_person_faces(dir):
    test_person_faces = []
    files = glob.glob(os.path.join(dir, "*.jpg"))
    for file in files:
        test_person_face = cv2.imread(file)
        test_person_face = cv2.cvtColor(test_person_face, cv2.COLOR_BGR2RGB)
        test_person_faces.append(test_person_face)
    return test_person_faces


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scanner_name",
        type=str,
        help="ID/Name of scanner.",
        default="hpaio:/usb/Deskjet_3070_B611_series?serial=CN19A371MR05MQ",
    )
    parser.add_argument(
        "--hp_sane_scanner",
        action="store_true",
        help="Use HP scanner supported by SANE. For more information see"
        + " https://gitlab.gnome.org/World/OpenPaperwork/pyinsane."
        + " Tested on Deskjet_3070_B611_series.",
    )
    parser.add_argument(
        "--gamma", type=float, help="Gamma of scanned image.", default=1.0
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="Save scanned document and recorded images of faces.",
    )
    parser.add_argument(
        "--data_save_dir",
        type=str,
        help="Path to directory, in which scanned document and recorded images "
        + "will be saved.",
        default="./scanned_documents/",
    )

    parser.add_argument(
        "--facenet_model_checkpoint_dir",
        type=str,
        help="Path to the facenet's model directory.",
        default="./model/facenet",
    )
    parser.add_argument(
        "--load_document_type_detector",
        action="store_true",
        help="Load document type detector.",
    )
    parser.add_argument(
        "--document_type_detector_means_path",
        type=str,
        help="Path to the pickle file with document type detector's means.",
        default="./models/document_type_detector_means.pkl",
    )

    parser.add_argument(
        "--min_nr_recorded_photos_treshold",
        type=int,
        help="Treshold for number of recorded photos of face.",
        default=40,
    )
    parser.add_argument(
        "--min_nr_similar_photos_treshold",
        type=int,
        help="Treshold for number of similar pairs of face's photographies "
        + "and document\s photography.",
        default=10,
    )
    parser.add_argument(
        "--min_distance",
        type=float,
        help="Minimum distance to detect similarity.",
        default=1.15,
    )
    parser.add_argument(
        "--recording_time",
        type=float,
        help="The number of seconds that recording will last.",
        default=10,
    )
    parser.add_argument(
        "--next_authorization_time",
        type=int,
        help="The number of seconds to the next authorization.",
        default=10,
    )

    parser.add_argument(
        "--realsense", action="store_true", help="Use Intel Realsense depth camera."
    )
    parser.add_argument(
        "--video_output_scale",
        type=int,
        help="Scale of video output during recording.",
        default=2,
    )
    parser.add_argument(
        "--video_width",
        type=int,
        help="Video width captured by camera in pixels.",
        default=640,
    )
    parser.add_argument(
        "--video_height",
        type=int,
        help="Video height captured by camera in pixels.",
        default=480,
    )
    parser.add_argument(
        "--frame_downscale",
        type=int,
        help="Scale of skipped pixels during bounding box detection.",
        default=2,
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        help="Number of skipped frames during face detection.",
        default=2,
    )
    parser.add_argument(
        "--check_depth_fraud",
        action="store_true",
        help="Check if fraud using depth camera.",
    )
    parser.add_argument(
        "--real_face_treshold",
        type=float,
        help="Minimum percentage of detected eyes' positions in real face area.",
        default=0.5,
    )
    parser.add_argument(
        "--real_person_treshold",
        type=float,
        help="Minimum percentage of real faces in detected images.",
        default=0.5,
    )

    parser.add_argument(
        "--image_size",
        type=int,
        help="Image size (height, width) in pixels.",
        default=160,
    )
    parser.add_argument(
        "--margin",
        type=int,
        help="Margin for the crop around the bounding box (height, width) in pixels.",
        default=44,
    )
    parser.add_argument(
        "--gpu_memory_fraction",
        type=float,
        help="Upper bound on the amount of GPU memory that will be used by the process.",
        default=1.0,
    )

    parser.add_argument(
        "--load_test_document_image",
        action="store_true",
        help="Load document image for testing (if you don't have scanner).",
    )
    parser.add_argument(
        "--test_document_image_path",
        type=str,
        help="Path to the document image for testing.",
        default="./scanned_documents/tests/document_image_test.jpg",
    )
    parser.add_argument(
        "--load_test_person_faces",
        action="store_true",
        help="Load test person's faces for testing (if you don't have camera).",
    )
    parser.add_argument(
        "--test_person_faces_path",
        type=str,
        help="Path to the directory with person's faces for testing.",
        default="./scanned_documents/tests/person_faces_test",
    )
    return parser.parse_args(argv)
