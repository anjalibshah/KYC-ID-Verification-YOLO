"""
App for serving TensorFlow models on Cloud Foundry/IBM Cloud.
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='kyc-demo-python',
    version='1.0.0',
    description='KYC ID verification app using pretrained Keras model.',
    long_description=long_description,
    license='Apache-2.0'
)
