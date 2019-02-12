#!/usr/bin/env python

try:
    from setuptools import setup

    _have_setuptools = True
except ImportError:
    from distutils.core import setup

setup(
    name="authorization_system",
    version="1.0",
    description="Authorization system",
    author="Piotr Smuda, Piotr Tempczyk",
    author_email="piotr.smuda@daftcode.pl",
    url="https://github.com/daftcode",
    packages=[
        "authorization_system",
        "authorization_system.document_processing",
        "authorization_system.face_recognition",
    ],
)
