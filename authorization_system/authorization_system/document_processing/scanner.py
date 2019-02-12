# coding=utf-8
"""
Scanner.
HP Deskjet 3070 B611 series and Fujitsu fi-65F were tested.
"""

import os
import pickle

import cv2
import imageio
import numpy as np
import pyinsane2
from authorization_system.document_processing import document, document_type_detector
from authorization_system.document_processing.document_utils import BColors
from authorization_system.face_recognition import face, face_utils


class DocumentScanner:
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
        if args.load_document_type_detector:
            # loading document type detector
            self.document_detector = document_type_detector.DocumentTypeDetector()
            self.document_detector.load_document_means(
                self.args.document_type_detector_means_path
            )
        else:
            self.document_detector = None

        if device is None and set_device:
            # setting scanner
            pyinsane2.init()
            device = pyinsane2.Scanner(name=args.scanner_name)
            device.options["mode"].value = "Color"
            device.options["resolution"].value = 600
            if args.hp_sane_scanner:
                device.options["tl-x"].value = 1 * 65536
                device.options["tl-y"].value = 1 * 65536
                device.options["br-x"].value = 84 * 65536
                device.options["br-y"].value = 51 * 65536
        self.scanner_device = device

    def _adjust_gamma(self, image, gamma=2.5):
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image.copy(), table)

    def scan_and_process(self, document_type=None, gamma=1.0, test_document_image=None):
        if test_document_image is None:
            # document scanning
            scan_session = self.scanner_device.scan(multiple=False)
            try:
                while True:
                    scan_session.scan.read()
            except EOFError:
                pass
            image = scan_session.images[-1]
            if self.args.hp_sane_scanner:
                document_image = np.array(image)
            else:
                # Remember to select proper values of cut (shape of (1180, 1960))
                # and "scanning margin". Every scanner device is different.
                # If you want to have solution insensitive on document flipping,
                # use any graphics software to make sure that you get similar
                # images after cut and flip. In our configuration margin is (24, 30).
                document_image = np.array(image)[30:1210, 24:1984]
            if gamma != 1.0:
                document_image = self._adjust_gamma(document_image, gamma)
        else:
            document_image = test_document_image.copy()

        # document processing: setting type; detecting first name and surname;
        #                      hiding sensitive data; detecting photo
        document_info = document.Document()
        document_info.process_document(
            document_image,
            type=document_type,
            document_type_detector=self.document_detector,
            face_recognition=self.face_recognition,
        )
        return document_info

    def scan_document(
        self,
        document_type=None,
        gamma=1.0,
        exit_scanner=True,
        save_data=False,
        intro=True,
        window_scan_title="Scanned Document",
        test_document_image=None,
    ):
        if intro:
            os.system("clear")
            print(
                "Welcome to the authorization system based on Polish documents.",
                "Place your document on the scanner, with the photograph facing",
                "downwards.",
            )
        print(
            "\nYour document is going to be scanned right now.",
            "After then it will be showed on the screen.",
        )

        # document scanning and processing
        document_info = self.scan_and_process(
            document_type=document_type,
            gamma=gamma,
            test_document_image=test_document_image,
        )

        # showing scanned document with hidden sensitive data
        document_image = cv2.cvtColor(document_info.image.copy(), cv2.COLOR_RGB2BGR)
        cv2.imshow(window_scan_title, document_image)
        cv2.waitKey(50)

        # saving data
        if save_data:
            self.scanner_device.save_document(document_info)
        if exit_scanner:
            self.scanner_device.exit()
        return document_info

    def check_if_correct_document(self, document_info):
        if document_info is None:
            return False
        elif document_info.type == "unknown_document":
            print(
                "\n%s%sUnknown document was detected. Try again.%s"
                % (BColors.BOLD, BColors.FAIL, BColors.ENDC)
            )
            return False
        elif document_info.photo is None:
            print(
                "\n%s%sPhoto on the document wasn't detected. Try again.%s"
                % (BColors.BOLD, BColors.FAIL, BColors.ENDC)
            )
            return False
        elif document_info.type_auto != document_info.type:
            print(
                (
                    "\n%s%sIncorrect document's type was detected after flipping."
                    + "Try again.%s"
                )
                % (BColors.BOLD, BColors.FAIL, BColors.ENDC)
            )
            return False
        elif (
            document_info.name_image is None
            or document_info.first_name_detected_text is None
            or document_info.surname_detected_text is None
        ):
            print(
                "\n%s%sText wasn't detected. Try again.%s"
                % (BColors.BOLD, BColors.FAIL, BColors.ENDC)
            )
            return False
        else:
            return True

    def save_document(self, document_info):
        if document_info is None:
            return None

        # creating paths
        dir = self.args.data_save_dir
        if not os.path.exists(os.path.join(dir, "full_document_info")):
            os.makedirs(os.path.join(dir, "full_document_info"))
        if not os.path.exists(os.path.join(dir, "document_image")):
            os.makedirs(os.path.join(dir, "document_image"))
        if not os.path.exists(os.path.join(dir, "document_type_image")):
            os.makedirs(os.path.join(dir, "document_type_image"))
        if not os.path.exists(os.path.join(dir, "document_photo")):
            os.makedirs(os.path.join(dir, "document_photo"))
        if not os.path.exists(os.path.join(dir, "document_name_image")):
            os.makedirs(os.path.join(dir, "document_name_image"))
        if not os.path.exists(os.path.join(dir, "document_box_name_image")):
            os.makedirs(os.path.join(dir, "document_box_name_image"))

        # saving files: all information about document in pickle; document's image;
        #               part of document for type detector; photo from document;
        #               part of document with first name and surname after processing;
        #               part of document with first name and surname before processing;
        document_info_file_name = document_info.timestamp + "_" + document_info.type
        document_info_file_name = document_info_file_name.replace(" ", "_")
        with open(
            os.path.join(dir, "full_document_info", document_info_file_name + ".pkl"),
            "wb",
        ) as f:
            pickle.dump(document_info, f, pickle.HIGHEST_PROTOCOL)
        if document_info.image is not None:
            imageio.imsave(
                os.path.join(dir, "document_image", document_info_file_name + ".jpg"),
                document_info.image,
            )
            imageio.imsave(
                os.path.join(
                    dir, "document_type_image", document_info_file_name + ".jpg"
                ),
                document_info.image[50:300, 50:650, :],
            )
        if document_info.photo is not None:
            imageio.imsave(
                os.path.join(dir, "document_photo", document_info_file_name + ".jpg"),
                document_info.photo,
            )
        if document_info.name_image is not None:
            imageio.imsave(
                os.path.join(
                    dir, "document_name_image", document_info_file_name + ".jpg"
                ),
                document_info.name_image,
            )
        if document_info.box_name_image is not None:
            imageio.imsave(
                os.path.join(
                    dir, "document_box_name_image", document_info_file_name + ".jpg"
                ),
                document_info.box_name_image,
            )

    def exit(self):
        pyinsane2.exit()
