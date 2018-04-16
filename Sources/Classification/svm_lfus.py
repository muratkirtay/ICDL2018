import numpy as np
import scipy.io as sio
import time
import logging as lg
import preprocessing as prep
import classification_funcs as cl

np.random.seed(42)
fname = 'logs/svm_lfus.txt'

desc = "Multi class SVM with spooled efus"
init = time.time()
lg.basicConfig(filename=fname, format='%(message)s', level=lg.INFO)
lg.info('Experiment date: %s', time.ctime())
lg.info('Experiment Description: %s', desc)

def main():

    nobjs, splen = 51, 1024*2
    io_path = '/home/neurobot/Datasets/classification_io/lfus/'
    lg.info('Data path: %s', io_path)
    # header is the same for all
    Xtr, Ytr = prep.extract_mat_content(io_path+'Xtr_lfus.mat', 'io'), prep.extract_mat_content(io_path+'Ytr_lfus.mat', 'io')
    Xval, Yval = prep.extract_mat_content(io_path+'Xval_lfus.mat', 'io'), prep.extract_mat_content(io_path+'Yval_lfus.mat', 'io')
    Xtest, Ytest = prep.extract_mat_content(io_path + 'Xtst_lfus.mat', 'io'), prep.extract_mat_content(io_path + 'Ytst_lfus.mat', 'io')

    io_info = 'Xtrain: ' + str(Xtr.shape) +' Ytrain: ' + str(Ytr.shape) +\
              ' Xval: ' + str(Xval.shape) + ' Yval: ' + str(Xval.shape) +\
              ' Xtest: ' + str(Xtest.shape) + ' Ytest: ' + str(Ytest.shape) + '\n'

    lg.info('IO shapes: %s', io_info)


    # Put them in classification_funcs.py
    Xtr_b, Xval_b, Xtest_b = cl.add_bias_nodes_to_inputs(Xtr, Xval, Xtest)
    W_b = np.random.randn(splen + 1, nobjs) * 0.00001

    niters, delta = 1000, 1.0
    estop_iter, estop_acc = 20, 0.01
    stepsize, reg = [1e-1, 1e-2, 1e-3, 1e-4, 2e-2, 2e-3, 2e-4], [1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5, 4e-5, 5e-5]

    for ssize in range(len(stepsize)):
        for regst in range(len(reg)):

            W_b = np.random.randn(splen + 1, nobjs) * 0.00001
            cv_flag, its = True, 0
            loss_mat, tr_acc, val_acc = [], [], []
            while cv_flag:
                loss, grad = cl.extract_loss_and_gradient(W_b, Xtr_b, Ytr[0], delta, reg[regst])
                W_b += -1 * stepsize[ssize] * grad

                tr_accuracy, trpred = cl.get_accuracy(Xtr_b, W_b, Ytr[0])
                val_accuracy, valpred = cl.get_accuracy(Xval_b, W_b, Yval[0])
                loss_mat.append(loss)
                tr_acc.append(tr_accuracy)
                val_acc.append(val_accuracy)
                print "%d/%d  Loss: %f Accuracy Training: %f Accuracy Validation: %f" %(its, niters, loss, tr_accuracy, val_accuracy)

                if its >= estop_iter:
                    loss_arr = val_acc[len(val_acc) - estop_iter: len(val_acc)]
                    if cl.early_stop(loss_arr, estop_acc) or its >= niters:
                        # lg.info("The validation accuracy is stable; break the loop for testing")
                        tst_accuracy = cl.get_accuracy(Xtest_b, W_b, Ytest[0])
                        tst_accuracy, Yte_pred = cl.get_accuracy(Xtest_b, W_b, Ytest)

                        Yactual_pred = np.vstack((Ytest[0], Yte_pred))
                        conf_mat = cl.construct_conf_mat(Ytest[0], Yte_pred, nobjs)

                        confname = '/home/neurobot/Datasets/results/lfus/confmat_' + str(ssize) + '_' + str(regst) + '.mat'
                        yactpred = '/home/neurobot/Datasets/results/lfus/yact_pred_' + str(ssize) + '_' + str(regst) + '.mat'
                        #weights = '/home/neurobot/Datasets/results/lfus/wb_' + str(ssize) + '_' + str(regst) + '.mat'
                        loss_trace = '/home/neurobot/Datasets/results/lfus/loss_' + str(ssize) + '_' + str(regst) + '.mat'
                        train_accs = '/home/neurobot/Datasets/results/lfus/tracc_' + str(ssize) + '_' + str(regst) + '.mat'
                        val_accs = '/home/neurobot/Datasets/results/lfus/vacc_' + str(ssize) + '_' + str(regst) + '.mat'

                        sio.savemat(confname, mdict={'conf_mat': conf_mat})
                        sio.savemat(yactpred, mdict={'yactpred': Yactual_pred})
                        #sio.savemat(weights, mdict={'weights': W_b})

                        sio.savemat(loss_trace, mdict={'loss': loss_mat})
                        sio.savemat(train_accs, mdict={'tr_accs': tr_acc})
                        sio.savemat(val_accs, mdict={'val_accs': val_acc})

                        lg.info("Learning rate: %E Regularization strength: %E, TR acc: %f, Val acc: %f Tst acc: %f iterations: %d" % (stepsize[ssize], reg[regst], tr_accuracy, val_accuracy, tst_accuracy, its))
                        cv_flag = False

                        print "=============================================="

                its += 1


if __name__ == '__main__':
    main()
    lg.info('Experiment Finished Date/Time: %s', time.ctime())