# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

setup(
    name='pypytorch',
    version='0.0.1',
    description='PyPyTorch is a simple deep learning framework',
    author='dark-ai',
    author_email='ac990731@gmail.com',
    platforms=['macOS', 'Windows', 'Linux'],
    url='https://www.github.com/dark-ai/pypytorch',
    packages=find_packages(),
    install_requires=['numpy', 'dill']
)
