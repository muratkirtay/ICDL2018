import numpy as np
import scipy.io as sio
import preprocessing as prep

np.random.seed(42)

def main():

    # balanced data path
    bspool_rgb = '/home/neurobot/Datasets/balanced_spool_washington_io/rgb_spool_b/'
    bspool_mask = '/home/neurobot/Datasets/balanced_spool_washington_io/mask_spool_b/'
    bspool_efus = '/home/neurobot/Datasets/balanced_spool_washington_io/mm_early_spool_b/'
    bspool_lfus = '/home/neurobot/Datasets/balanced_spool_washington_io/mm_late_spool_b/'

    srgb_header, smask_header, slfusion_header, sefusion_header = 'srgb', 'smask', 'slfusion', 'sefusion'

    rtrain, rval, rtest = 248, 123, 123
    nobjs, min_size = 51, 494 # prep.get_min_size(data_path, data_header)
    lmodal, lmodalities = 1024, 2048

    # dtype: X-> np.float32, Y-> np.int32
    # use np.repeat to generate Y

    # training
    Xtr_rgb, Xtr_mask = np.zeros((nobjs*rtrain, lmodal), dtype=np.float32), np.zeros((nobjs*rtrain, lmodal), dtype=np.float32)
    Xtr_efus, Xtr_lfus = np.zeros((nobjs*rtrain, lmodalities), dtype=np.float32), np.zeros((nobjs*rtrain, lmodalities), dtype=np.float32)

    Ytr_rgb, Ytr_mask = np.zeros((nobjs*rtrain), dtype=np.int32), np.zeros((nobjs*rtrain), dtype=np.int32)

    Ytr_efus, Ytr_lfus = np.zeros((nobjs*rtrain), dtype=np.int32), np.zeros((nobjs*rtrain), dtype=np.int32)

    # validation
    Xval_rgb, Xval_mask = np.zeros((nobjs*rval, lmodal), dtype=np.float32), np.zeros((nobjs*rval, lmodal), dtype=np.float32)
    Xval_efus, Xval_lfus = np.zeros((nobjs*rval, lmodalities), dtype=np.float32), np.zeros((nobjs*rval, lmodalities), dtype=np.float32)

    Yval_rgb, Yval_mask = np.zeros((nobjs*rval), dtype=np.int32), np.zeros((nobjs*rval), dtype=np.int32)
    Yval_efus, Yval_lfus = np.zeros((nobjs*rval), dtype=np.int32), np.zeros((nobjs*rval), dtype=np.int32)

    # testing
    Xtest_rgb, Xtest_mask = np.zeros((nobjs*rtest, lmodal), dtype=np.float32), np.zeros((nobjs*rtest, lmodal), dtype=np.float32)
    Xtest_efus, Xtest_lfus = np.zeros((nobjs*rtest, lmodalities), dtype=np.float32), np.zeros((nobjs*rtest, lmodalities), dtype=np.float32)

    Ytest_rgb, Ytest_mask = np.zeros((nobjs*rtest), dtype=np.int32), np.zeros((nobjs*rtest), dtype=np.int32)
    Ytest_efus, Ytest_lfus = np.zeros((nobjs*rtest), dtype=np.int32), np.zeros((nobjs*rtest), dtype=np.int32)

    # I/O generation *must* be in the same loop, otherwise there will be mismatch among indices
    for i in range(nobjs):
        obj_str = str(i+1)
        data_rgb = prep.extract_mat_content(bspool_rgb + obj_str + '.mat', srgb_header)
        data_mask = prep.extract_mat_content(bspool_mask + obj_str + '.mat', smask_header)
        data_efus = prep.extract_mat_content(bspool_efus + obj_str + '.mat', sefusion_header)
        data_lfus = prep.extract_mat_content(bspool_lfus + obj_str + '.mat', slfusion_header)

        ids = np.random.permutation(np.arange(min_size))
        id_train, id_val, id_test = ids[0:rtrain], ids[rtrain:rtrain + rval], ids[rtrain + rval:]

        # RGB I/O mats
        Xtr_rgb[rtrain*i: i*rtrain+rtrain], Ytr_rgb[rtrain*i: i*rtrain+rtrain] = data_rgb[id_train], np.repeat(i, rtrain)
        Xval_rgb[rval*i: i*rval+rval], Yval_rgb[rval*i: i*rval+rval] = data_rgb[id_val], np.repeat(i, rval)
        Xtest_rgb[rtest*i: i*rtest+rtest], Ytest_rgb[rtest*i: i*rtest+rtest] = data_rgb[id_test], np.repeat(i, rtest)

        # Mask I/O mats
        Xtr_mask[rtrain*i: i*rtrain+rtrain], Ytr_mask[rtrain*i: i*rtrain+rtrain] = data_mask[id_train], np.repeat(i, rtrain)
        Xval_mask[rval*i: i*rval+rval], Yval_mask[rval*i: i*rval+rval] = data_mask[id_val], np.repeat(i, rval)
        Xtest_mask[rtest*i: i*rtest+rtest], Ytest_mask[rtest*i: i*rtest+rtest] = data_mask[id_test], np.repeat(i, rtest)

        # Efusion I/O mats
        Xtr_efus[rtrain * i: i * rtrain + rtrain], Ytr_efus[rtrain * i: i * rtrain + rtrain] = data_efus[id_train], np.repeat(i, rtrain)
        Xval_efus[rval * i: i * rval + rval], Yval_efus[rval * i: i * rval+ rval] = data_efus[id_val], np.repeat(i, rval)
        Xtest_efus[rtest * i: i * rtest + rtest], Ytest_efus[rtest * i: i * rtest + rtest] = data_efus[id_test], np.repeat(i, rtest)

        # Lfusion I/O mats
        Xtr_lfus[rtrain * i: i * rtrain + rtrain], Ytr_lfus[rtrain * i: i * rtrain + rtrain] = data_lfus[id_train], np.repeat(i, rtrain)
        Xval_lfus[rval * i: i * rval + rval], Yval_lfus[rval * i: i * rval + rval] = data_lfus[id_val], np.repeat(i, rval)
        Xtest_lfus[rtest * i: i * rtest + rtest], Ytest_lfus[rtest * i: i * rtest + rtest] = data_lfus[id_test], np.repeat(i, rtest)


    io_path = '/home/neurobot/Datasets/classification_io/'
    # rgb mats
    Xrgb_tr, Yrgb_tr = io_path + 'rgb/Xtr_rgb.mat',  io_path + 'rgb/Ytr_rgb.mat'
    Xrgb_val, Yrgb_val = io_path + 'rgb/Xval_rgb.mat', io_path + 'rgb/Yval_rgb.mat'
    Xrgb_tst, Yrgb_tst = io_path + 'rgb/Xtst_rgb.mat', io_path + 'rgb/Ytst_rgb.mat'

    sio.savemat(Xrgb_tr, mdict={'io': Xtr_rgb}), sio.savemat(Yrgb_tr, mdict={'io': Ytr_rgb})
    sio.savemat(Xrgb_val, mdict={'io': Xval_rgb}), sio.savemat(Yrgb_val, mdict={'io': Yval_rgb})
    sio.savemat(Xrgb_tst, mdict={'io': Xtest_rgb}), sio.savemat(Yrgb_tst, mdict={'io': Ytest_rgb})

    # mask mats
    Xmask_tr, Ymask_tr = io_path + 'mask/Xtr_mask.mat', io_path + 'mask/Ytr_mask.mat'
    Xmask_val, Ymask_val = io_path + 'mask/Xval_mask.mat', io_path + 'mask/Yval_mask.mat'
    Xmask_tst, Ymask_tst = io_path + 'mask/Xtst_mask.mat', io_path + 'mask/Ytst_mask.mat'

    sio.savemat(Xmask_tr, mdict={'io': Xtr_mask}), sio.savemat(Ymask_tr, mdict={'io': Ytr_mask})
    sio.savemat(Xmask_val, mdict={'io': Xval_mask}), sio.savemat(Ymask_val, mdict={'io': Yval_mask})
    sio.savemat(Xmask_tst, mdict={'io': Xtest_mask}), sio.savemat(Ymask_tst, mdict={'io': Ytest_mask})

    # efusion mats
    Xefus_tr, Yefus_tr = io_path + 'efus/Xtr_efus.mat', io_path + 'efus/Ytr_efus.mat'
    Xefus_val, Yefus_val = io_path + 'efus/Xval_efus.mat', io_path + 'efus/Yval_efus.mat'
    Xefus_tst, Yefus_tst = io_path + 'efus/Xtst_efus.mat', io_path + 'efus/Ytst_efus.mat'

    sio.savemat(Xefus_tr, mdict={'io': Xtr_efus}), sio.savemat(Yefus_tr, mdict={'io': Ytr_efus})
    sio.savemat(Xefus_val, mdict={'io': Xval_efus}), sio.savemat(Yefus_val, mdict={'io': Yval_efus})
    sio.savemat(Xefus_tst, mdict={'io': Xtest_efus}), sio.savemat(Yefus_tst, mdict={'io': Ytest_efus})

    # lfusion mats
    Xlfus_tr, Ylfus_tr = io_path + 'lfus/Xtr_lfus.mat', io_path + 'lfus/Ytr_lfus.mat'
    Xlfus_val, Ylfus_val = io_path + 'lfus/Xval_lfus.mat', io_path + 'lfus/Yval_lfus.mat'
    Xlfus_tst, Ylfus_tst = io_path + 'lfus/Xtst_lfus.mat', io_path + 'lfus/Ytst_lfus.mat'

    sio.savemat(Xlfus_tr, mdict={'io': Xtr_lfus}), sio.savemat(Ylfus_tr, mdict={'io': Ytr_lfus})
    sio.savemat(Xlfus_val, mdict={'io': Xval_lfus}), sio.savemat(Ylfus_val, mdict={'io': Yval_lfus})
    sio.savemat(Xlfus_tst, mdict={'io': Xtest_lfus}), sio.savemat(Ylfus_tst, mdict={'io': Ytest_lfus})


if __name__ == '__main__':
    main()