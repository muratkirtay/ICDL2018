import numpy as np
from mHTM.region import SPRegion
import scipy.io as sio
import processing as hl

seed = 123456789

def main():

    print "RGB SDRs processing"
    hl.tic()
    # Parameters to construct cortical structure
    nbits, pct_active, nobjs = 1024, 0.4, 51
    nofcols, nofcols_ef = 1024, 2048

    # Binary path
    bin_path_rgb = '/home/neurobot/Datasets/mmodal_washington_io/rgb/'
    bin_path_mask = '/home/neurobot/Datasets/mmodal_washington_io/mask/'
    bin_path_efussion = '/home/neurobot/Datasets/mmodal_washington_io/efusion/'
    brgb_header, bmask_header, befusion_header = 'brgb', 'bmask', 'befusion'
    srgb_header, smask_header, sefusion_header = 'srgb', 'smask', 'sefusion'

    # Spool path
    spool_rgb = '/home/neurobot/Datasets/spool_washington_io/rgb_spool/'
    spool_mask = '/home/neurobot/Datasets/spool_washington_io/mask_spool/'
    spool_efus = '/home/neurobot/Datasets/spool_washington_io/mm_early_spool/'
    spool_lfus = '/home/neurobot/Datasets/spool_washington_io/mm_late_spool/'

    kargs = {'ninputs': nbits, 'ncolumns': nofcols, 'nactive': int(nbits * 0.2),
             'global_inhibition': True, 'trim': 1e-4, 'disable_boost': True,
             'seed': seed, 'nsynapses': 100, 'seg_th': 10, 'syn_th': 0.5,
             'pinc': 0.001, 'pdec': 0.001, 'pwindow': 0.5, 'random_permanence': True,
            'nepochs': 10 }

    sp = SPRegion(**kargs)

    # Change the path according to modality type
    data_path, data_header = bin_path_rgb, brgb_header
    sdr_path, sdr_header = spool_rgb, srgb_header

    for j in range(nobjs):
        obj_str = str(j+1)
        obj_path = data_path + obj_str + '.mat'
        data_content = hl.extract_mat_content(obj_path, data_header)
        num_of_imgs, length = data_content.shape
        sdrs = np.zeros((num_of_imgs, nbits), dtype = np.int64)
        for i in range(num_of_imgs):
            sp.fit(data_content[i])
            sp_output = sp.predict(data_content[i])
            outp = sp_output * 1
            if np.count_nonzero(outp) != int(nbits * 0.2):
                print j+1, i, np.count_nonzero(outp)
            sdrs[i, :] = outp
        sdr_mat = sdr_path + str(j+1) + '.mat'
        sio.savemat(sdr_mat, mdict={sdr_header: sdrs})
        print "Spooling for object ", j+1, " tooks"
        hl.tac()
        print "----------------------------------"
    print "Finished spooling for all objects"
    hl.tac()

if __name__ == '__main__':
    main()
