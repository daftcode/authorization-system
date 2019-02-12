# coding=utf-8
"""
Authorization system.
To see usage run this file with '--help' argument.
"""

import os
import signal
import sys
import time
from termios import tcflush, TCIFLUSH

import cv2
import numpy as np
from authorization_system.document_processing import scanner, document_utils
from authorization_system.document_processing.document import TYPE_MAP
from authorization_system.face_recognition import (
    depth_antifraud,
    face,
    face_recorder,
    face_utils,
)


def signal_handler(sig, frame):
    cv2.destroyAllWindows()
    sys.exit(0)


def bool_to_string(value):
    if value:
        return "Yes"
    else:
        return "No"


def none_to_string(value):
    if value is None:
        return ""
    else:
        return value


class AuthorizationSystem:
    def __init__(self, args, face_recognition=None, device=None, set_device=True):
        self.args = args
        if face_recognition is None:
            face_utils.set_face_model(args, verbose=False)
            face_recognition = face.Recognition(
                face_crop_size=args.image_size,
                face_crop_margin=args.margin,
                gpu_memory_fraction=args.gpu_memory_fraction,
            )
        self.face_recognition = face_recognition
        self.scanner_device = scanner.DocumentScanner(
            args, self.face_recognition, device, set_device
        )
        self.recorder = face_recorder.FaceRecorder(args, self.face_recognition)
        self.antifraud = depth_antifraud.DepthAntifraud(args)
        self.window_recording_title = "Face Recording"
        self.window_scanned_title = "Scanned Document"
        self.empty_record = np.zeros((610, 1084), dtype=int)
        self.empty_scan = np.zeros((383, 636), dtype=int)

    def show_windows(self):
        cv2.namedWindow(self.window_recording_title, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.window_recording_title, 1084, 610)
        cv2.moveWindow(self.window_recording_title, 0, 0)
        cv2.imshow(self.window_recording_title, self.empty_record)
        cv2.waitKey(50)
        cv2.namedWindow(self.window_scanned_title, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.window_scanned_title, 1084, 383)
        cv2.moveWindow(self.window_scanned_title, 0, 667)
        cv2.imshow(self.window_scanned_title, self.empty_scan)
        cv2.waitKey(50)

    def scan_document(self, gamma, test_document_image=None):
        try:
            return self.scanner_device.scan_document(
                gamma=gamma,
                exit_scanner=False,
                intro=False,
                window_scan_title=self.window_scanned_title,
                test_document_image=test_document_image,
            )
        except:
            print(
                "\n%s%sScanner error!%s"
                % (
                    document_utils.BColors.BOLD,
                    document_utils.BColors.FAIL,
                    document_utils.BColors.ENDC,
                )
            )
            return None

    def check_document(self, document_info):
        return self.scanner_device.check_if_correct_document(document_info)

    def record_faces(self, test_person_faces=None):
        try:
            return self.recorder.record_face(
                recording_time=self.args.recording_time,
                scale=self.args.video_output_scale,
                width=self.args.video_width,
                height=self.args.video_height,
                frame_downscale=self.args.frame_downscale,
                frame_interval=self.args.frame_interval,
                realsense=self.args.realsense,
                window_recording_title=self.window_recording_title,
                test_person_faces=test_person_faces,
                verbose=False,
            )
        except:
            print(
                "\n%s%sRecording error!%s"
                % (
                    document_utils.BColors.BOLD,
                    document_utils.BColors.FAIL,
                    document_utils.BColors.ENDC,
                )
            )
            return None, None

    def check_recording(self, detected_faces):
        return self.recorder.check_if_correct_recording(
            detected_faces, self.args.min_nr_recorded_photos_treshold
        )

    def check_if_recognised(self, document_info, detected_faces, detected_depth_faces):
        if self.args.check_depth_fraud:
            real_person, _ = self.antifraud.check_depth_faces(detected_depth_faces)
            document_info.fraud_try = not real_person
        document_info, _ = self.recorder.check_if_recognised(
            document_info=document_info,
            detected_faces=detected_faces,
            min_distance=self.args.min_distance,
            min_nr_similar_photos_treshold=self.args.min_nr_similar_photos_treshold,
            verbose=False,
        )
        return document_info

    def print_results(self, document_info):
        if document_info is not None:
            summary_info = (
                TYPE_MAP[document_info.type],
                none_to_string(document_info.first_name_detected_text),
                none_to_string(document_info.surname_detected_text),
                bool_to_string(document_info.recognised),
            )
            summary_fraud = bool_to_string(document_info.fraud_try)
        else:
            summary_info = (
                TYPE_MAP["unknown_document"],
                none_to_string(None),
                none_to_string(None),
                bool_to_string(False),
            )
            summary_fraud = bool_to_string(False)

        print(
            (
                "\nSummary:\n\nDocument type: %s\nDetected first name: %s"
                + "\nDetected surname: %s\nRecognised: %s"
            )
            % summary_info
        )
        if self.args.check_depth_fraud:
            print("Fraud:", summary_fraud)

    def save_recording(self, faces, depth_faces, document_info):
        self.recorder.save_recording(faces, depth_faces, document_info)

    def save_document(self, document_info):
        self.scanner_device.save_document(document_info)


def authorization(args):
    if args.load_test_document_image:
        set_device = False
        test_document_image = document_utils.load_test_document_image(
            args.test_document_image_path
        )
    else:
        set_device = True
        test_document_image = None
    if args.load_test_person_faces:
        test_person_faces = document_utils.load_test_person_faces(
            args.test_person_faces_path
        )
    else:
        test_person_faces = None
    authorization_system = AuthorizationSystem(args, set_device=set_device)
    signal.signal(signal.SIGINT, signal_handler)
    while True:
        os.system("clear")
        print(
            "%s%sWelcome to the authorization system based on Polish documents.%s"
            % (
                document_utils.BColors.BOLD,
                document_utils.BColors.OKBLUE,
                document_utils.BColors.ENDC,
            )
        )
        authorization_system.show_windows()
        tcflush(sys.stdin, TCIFLUSH)
        input(
            (
                "\n%s%sPlace your document on the scanner, with the photograph facing"
                + " downwards and press 'enter'.%s"
            )
            % (
                document_utils.BColors.BOLD,
                document_utils.BColors.WARNING,
                document_utils.BColors.ENDC,
            )
        )
        document_info = authorization_system.scan_document(
            gamma=args.gamma, test_document_image=test_document_image
        )
        correct_document = authorization_system.check_document(document_info)
        if correct_document:
            detected_faces, detected_depth_faces = authorization_system.record_faces(
                test_person_faces=test_person_faces
            )
            correct_recording = authorization_system.check_recording(detected_faces)
            if correct_recording:
                print(
                    "\nYour document is being compared with recorded images right now."
                )
                document_info = authorization_system.check_if_recognised(
                    document_info, detected_faces, detected_depth_faces
                )
                if args.save_data:
                    authorization_system.save_recording(
                        detected_faces, detected_depth_faces, document_info
                    )
        if args.save_data:
            authorization_system.save_document(document_info)
        authorization_system.print_results(document_info)
        print(
            (
                "\n%s%sRemember to take your document!%s\n\n"
                + "Next authorization will be possible in %d seconds."
            )
            % (
                document_utils.BColors.BOLD,
                document_utils.BColors.FAIL,
                document_utils.BColors.ENDC,
                args.next_authorization_time,
            )
        )
        time.sleep(args.next_authorization_time)


if __name__ == "__main__":
    authorization(document_utils.parse_arguments(sys.argv[1:]))
