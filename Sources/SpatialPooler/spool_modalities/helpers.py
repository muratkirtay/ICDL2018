import numpy as np
import cv2
import scipy.io as sio
import time

_start_time = time.time()

def tic():
    global _start_time
    _start_time = time.time()

def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour,t_min) = divmod(t_min, 60)
    print t_hour, " hour: ", t_min, " mins: ", t_sec, " sec"

def extract_mat_content(path, header='obj'):
    """ Extract content of a .mat file located in path and has unique header.
    """
    content = sio.loadmat(path)

    return content[header]
