# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------
from .imdb import imdb
from . import factory

import os.path as osp
ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')

