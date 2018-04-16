import numpy as np
import scipy.io as sio


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


np.random.seed(42)

def main():

    spool_rgb = '/home/neurobot/Datasets/spool_washington_io/rgb_spool/'
    spool_mask = '/home/neurobot/Datasets/spool_washington_io/mask_spool/'
    spool_efus = '/home/neurobot/Datasets/spool_washington_io/mm_early_spool/'
    spool_lfus = '/home/neurobot/Datasets/spool_washington_io/mm_late_spool/'

    # balanced data path
    bspool_rgb = '/home/neurobot/Datasets/balanced_spool_washington_io/rgb_spool_b/'
    bspool_mask = '/home/neurobot/Datasets/balanced_spool_washington_io/mask_spool_b/'
    bspool_efus = '/home/neurobot/Datasets/balanced_spool_washington_io/mm_early_spool_b/'
    bspool_lfus = '/home/neurobot/Datasets/balanced_spool_washington_io/mm_late_spool_b/'

    srgb_header, smask_header, slfusion_header, sefusion_header = 'srgb', 'smask', 'slfusion', 'sefusion'

    nobjs = 51
    data_path, data_header = spool_rgb, srgb_header
    min_size = 494  # get_min_size(data_path, data_header)

    for i in range(nobjs):
        obj_str = str(i+1)

        data_rgb = extract_mat_content(spool_rgb + obj_str + '.mat', srgb_header)
        data_mask = extract_mat_content(spool_mask + obj_str + '.mat', smask_header)
        data_efus = extract_mat_content(spool_efus + obj_str + '.mat', sefusion_header)
        data_lfus = extract_mat_content(spool_lfus + obj_str + '.mat', slfusion_header)

        # print data_rgb.shape[0], data_mask.shape[0], data_efus.shape[0], data_lfus.shape[0]
        dsize = data_rgb.shape[0]
        # rand_inds = np.random.randint(dsize, size=min_size) # this is the repeated one, use randperm
        rand_inds = np.random.permutation(np.arange(dsize))[:min_size]

        brgb_sdr, bmask_sdr = bspool_rgb + str(i+1) + '.mat', bspool_mask + str(i+1) + '.mat'
        befus_sdr, blfus_sdr = bspool_efus + str(i+1) + '.mat', bspool_lfus + str(i+1) + '.mat'

        sio.savemat(brgb_sdr, mdict={srgb_header: data_rgb[rand_inds]})
        sio.savemat(bmask_sdr, mdict={smask_header: data_mask[rand_inds]})
        sio.savemat(befus_sdr, mdict={sefusion_header: data_efus[rand_inds]})
        sio.savemat(blfus_sdr, mdict={slfusion_header: data_lfus[rand_inds]})


if __name__ == '__main__':
    main()
