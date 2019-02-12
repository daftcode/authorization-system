![](https://img.shields.io/badge/python-3.6-blue.svg)
![](https://img.shields.io/badge/code%20style-black-000000.svg)

# Authorization System

Person identification based on scan of Polish document and recorded images by camera. 
[David's Sandberg facenet implementation](https://github.com/davidsandberg/facenet) 
was used for face recognition.

The code was tested on Ubuntu 18.04 LTS with HP 3070A/Fujitsu fi-65F (scanners with 600 
DPI) and front camera in DELL XPS/Intel RealSense Depth Camera D435.

## Dependencies

First of all, it is needed to install Tesseract. It's installation tutorial can be found 
[here](https://www.pyimagesearch.com/2017/07/03/installing-tesseract-for-ocr/). For 
Polish letters remember to install `tesseract-ocr-pol`. We used `tesseract 4.0.0-beta.1`
version.

Some packages are required before installation our library:

```
numpy==1.15.1
scipy==1.1.0
imageio==2.4.1
opencv-python==3.4.2.17
pytesseract==0.2.4
pyinsane2==2.0.13
pyrealsense2==2.14.1.91
tensorflow==1.7.0
facenet==1.0.3
```

You can install it with `pip install -r requirements.txt`. If you have GPU-enabled PC 
you can use `tensorflow-gpu==1.8.0` or greater. 

We recommend to use `virtualenv` with Python 3.6.X (3.6.5 or greater).

Required FaceNet model can be downloaded from 
[here](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view).
This model was trained on [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
dataset.

## Installation

You can install authorization system library in two ways.

Using `wheel` file:

```
pip install authorization_system-*.whl
```

or being within folder with `setup.py`:

```
pip install -e .
```

## Usage

To launch authorization system use:

```
python path_to_folder_containing_file/authorization.py [args]
```

To see all options for `authorization.py` use:

```
python path_to_folder_containing_file/authorization.py --help
```

Examples:

```
python ./authorization.py 
```

```
python ./authorization.py --facenet_model_checkpoint_dir ../models/facenet/ --document_type_detector_means_path ../models/document_type_detector_means.pkl
```

```
python ./authorization.py --facenet_model_checkpoint_dir ../models/facenet/ --document_type_detector_means_path ../models/document_type_detector_means.pkl --save_data --data_save_dir ../scanned_documents/ --realsense --video_output_scale 1 --video_width 1280 --video_height 720 
```

To quit press `Ctrl + C` in the console.

## Data

Attached data contains examples of two scanned documents and templates for document type
detector. Data is enscrypted and a password is `IA_edoctfaD` (written backwards).

## Remarks

Every scanner device is different, so don't worry if something go wrong on the first 
try. Correct working of our solution depends on a proper cutting of the document from 
scanned image. Therefore, it is important to set a good position of the document's area.
The area of the cut or final image should have 1180x1960px. 

If you want to have solution insensitive to document flipping, use any graphics software 
to make sure that you get similar images after cutting off and flipping (with very small 
shifts).

Due to the quality of different scanners, you can adjust gamma of the scans. It can be 
helpful, if there is any problem with document's type detection.

The resolution of video captured by the camera is set up to 640x480px and scale equaled 
to 2 by default. If you use Intel RealSense Depth Camera D435 you should set up it to 
1280x720px with scale equaled to 1. Remember to set scale correctly, because this 
parameter is also used for displaying bounding-boxes.

Fraud detection is sensitive to the face's position. If you try to "hide" face, it 
should be detected as fraud. Therefore, keep face straight to the camera.

## Authors

* Piotr Smuda
* Piotr Tempczyk