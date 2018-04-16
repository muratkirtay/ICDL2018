import numpy as np
from mHTM.region import SPRegion
import scipy.io as sio
import processing as hl

seed = 123456789

def main():

    print "Late fusion SDRs processing"
    hl.tic()
    # Parameters to construct cortical structure
    nbits, pct_active, nobjs = 2048, 0.4, 51

    # Spool path
    spool_rgb = '/home/neurobot/Datasets/spool_washington_io/rgb_spool/'
    spool_mask = '/home/neurobot/Datasets/spool_washington_io/mask_spool/'
    sdr_path = '/home/neurobot/Datasets/spool_washington_io/mm_late_spool/'
    # late fusion path
    srgb_header, smask_header, lfusion_header = 'srgb', 'smask', 'slfusion'

    for j in range(nobjs):
        obj_str = str(j+1)
        rgb_path = spool_rgb + obj_str + '.mat'
        mask_path = spool_mask + obj_str + '.mat'
        data_content_rgb = hl.extract_mat_content(rgb_path, srgb_header)
        data_content_mask = hl.extract_mat_content(mask_path, smask_header)

        num_of_imgs, length = data_content_rgb.shape
        mm_data = np.zeros((num_of_imgs, length*2), dtype = np.int64)

        for i in range(num_of_imgs):
            mm_data_rep = np.append(data_content_rgb[i], data_content_mask[i])
            mm_data[i, :] = mm_data_rep
        sdr_mat = sdr_path + str(j+1) + '.mat'
        sio.savemat(sdr_mat, mdict={lfusion_header: mm_data})
        print "MModaling for object ", j+1, " tooks"
        hl.tac()
        print "----------------------------------"
    print "Finished MModaling for all objects"
    hl.tac()

if __name__ == '__main__':
    main()
