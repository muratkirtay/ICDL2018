import numpy as np
import scipy.io as sio
import cv2
import os
import re
import time
import preprocessing as prep

_start_time = time.time()

def main():

    prep.tic()

    print "Binarization process for rgb and dmask took: "

    rsize, inplen, nclasses, nobjs, nimages = (32, 32), 32*32, 51, 300, 250000
    dataset_path = '/home/neurobot/Datasets/washington/rgbd-dataset/'
    bin_path_rgb = '/home/neurobot/Datasets/mmodal_washington_io/rgb/'
    bin_path_mask = '/home/neurobot/Datasets/mmodal_washington_io/mask/'
    bin_path_efussion = '/home/neurobot/Datasets/mmodal_washington_io/efusion/'
    # bin_path_depth = '/home/neurobot/Datasets/mmodal_washington_io/depth/'
    brgb_header, bmask_header, bdepth_header, befusion_header = 'brgb', 'bmask', 'bdepth', 'befusion'

    dlist = sorted(os.listdir(dataset_path))

    for i in range(len(dlist)):
        subfolds = dataset_path + dlist[i] +'/'
        dsublist = sorted(os.listdir(subfolds))
        classid = i +1

        for ii in range(len(dsublist)):
            flist = dataset_path + dlist[i] +'/' + dsublist[ii] + '/'
            fimglist = sorted(os.listdir(flist))

            reg_rgb, reg_mask, reg_depth = re.compile('.*_crop.png'), re.compile('.*_maskcrop.png'), re.compile('.*_depthcrop.png')
            rgb_imgs, mask_imgs, depth_imgs = filter(reg_rgb.match, fimglist), filter(reg_mask.match, fimglist), filter(reg_depth.match, fimglist)
            len_rgb, len_mask, len_depth = len(rgb_imgs), len(mask_imgs), len(depth_imgs)

            # construct raw input matrices before spooling
            xbin_rgb = np.zeros((len_rgb, inplen), dtype=np.float32)
            xbin_mask = np.zeros((len_rgb, inplen), dtype=np.float32)
            xbin_efusion = np.zeros((len_rgb, inplen * 2), dtype=np.float32)
            # xbin_depth = np.zeros((len_rgb, inplen), dtype=np.float32)

          
            for j in range(len(rgb_imgs)):
                rgbname = flist + rgb_imgs[j]
                maskname = rgbname[0 : len(rgbname)-8] +'maskcrop.png'

                # process RGB
                rimg, gimg  = cv2.imread(rgbname), cv2.imread(rgbname,  cv2.CV_LOAD_IMAGE_GRAYSCALE)
                brimg = cv2.adaptiveThreshold(gimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 151, 13)
                rsized_brimg = cv2.resize(brimg, rsize)
                nonz = rsized_brimg.nonzero()
                rsized_brimg[nonz] = 1.0
                rbin_vector = rsized_brimg.flatten()

                # process depth mask
                mimg = cv2.imread(maskname, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                rsized_mask = cv2.resize(mimg, rsize)
                mnonz = rsized_mask.nonzero()
                rsized_mask[mnonz] = 1.0
                maskbin_vector = rsized_mask.flatten()
                # early fusion
                efussed_modalities = np.append(rbin_vector, maskbin_vector)

                xbin_rgb[j, :] = rbin_vector
                xbin_mask[j, :] = maskbin_vector
                xbin_efusion[j, :] = efussed_modalities

            brgb_mat = bin_path_rgb + str(classid) + '.mat'
            bmask_mat = bin_path_mask + str(classid) + '.mat'
            befusion_mat = bin_path_efussion + str(classid) + '.mat'

            sio.savemat(brgb_mat, mdict={brgb_header: xbin_rgb})
            sio.savemat(bmask_mat, mdict={bmask_header: xbin_mask})
            sio.savemat(befusion_mat, mdict={befusion_header: xbin_efusion})

            xbin_rgb = np.zeros((len_rgb, inplen), dtype=np.float32)
            xbin_mask = np.zeros((len_rgb, inplen), dtype=np.float32)
            xbin_efusion = np.zeros((len_rgb, inplen * 2), dtype=np.float32)

    prep.tac()


if __name__ == '__main__':
    main()