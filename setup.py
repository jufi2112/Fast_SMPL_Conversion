# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import re

from os import path as osp
from setuptools import setup, find_packages

def read_version():
    with open(osp.join('smpl_conversion', 'version.py'), 'rt') as file:
        version_file = file.read()
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError(f"Could not find version string in {osp.join('smpl_conversion', 'version.py')}")

setup(
    name='smpl_conversion',
    version=read_version(),
    packages=find_packages(),
    description='Conversion of Parameterized Human Body Models Using Fully-Connected Neural Networks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Julien Fischer',
    author_email='julien.fischer@tu-dresden.de',
    url='https://jufi2112.github.io/Fast_SMPL_Conversion/',
    install_requires=[
        'torch==1.12.1',
        'dotmap',
        'matplotlib',
        'mkl==2024.0.0',
        'numpy==1.23.1',
        'numpy-quaternion',
        'scikit-learn',
        'scipy',
        'tqdm',
        'trimesh',
        'vedo',
        'pyyaml',
        'chumpy==0.70',
        'omegaconf',
        'open3d==0.17.0',
        'npy_append_array',
        'torch_trust_ncg',
        'smplx',
        'supr',
        'star'
    ],
    python_requires='>=3.9'
)
