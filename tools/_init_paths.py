# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

"""Set up paths for AMT-GAF"""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)


# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'python')
add_path(lib_path)
