import numpy as np
import cv2
from pathlib import Path
import time
import os
import scipy.io as sio


# to be consistent with different versions of the opencv
cv2.CV_LOAD_IMAGE_GRAYSCALE = 0


def tic():
    """ Log starting time to run specific code block."""
    global _start_time
    _start_time = time.time()


def tac():
    """ Print logged time in hour : min : sec format """

    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour,t_min) = divmod(t_min, 60)

    print t_hour, " hour: ", t_min, " mins: ", t_sec, " sec"


def check_file_exists(dmaskcrop, fname):
    """ Move the files that has no segmentation mask"""

    rm_path = ' /home/neurobot/Datasets/mmodal_washington_io/removed_files/'
    dfile = Path(dmaskcrop)

    if not dfile.is_file():
        cmd = 'mv ' + fname + "*" + rm_path
        os.system(cmd)


def extract_mat_content(path, header='obj'):
    """ Extract content of a .mat file located in path and has unique header.
    """
    content = sio.loadmat(path)

    return content[header]


def get_min_size(data_path, data_header):
    """ Get the mininum number of images of all objects"""

    nobjs, size_list = 51, []

    for i in range(nobjs):
        obj_str = str(i+1)
        obj_path = data_path + obj_str + '.mat'
        data_content = extract_mat_content(obj_path, data_header)

        size_list.append(data_content.shape[0])

    return min(size_list)