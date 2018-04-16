import numpy as np
import preprocessing as prep

def extract_loss_and_gradient(W_b, Xtr_b, Ytr, delta, reg):
    """ Perform gradient descent to optimize Hing-loss.
        Bias nodes included to regularization procedure.
    """
    loss = 0.0
    dW = np.zeros(W_b.shape)

    ntraining = Xtr_b.shape[0]
    fscores = Xtr_b.dot(W_b)

    ftrue = fscores[np.arange(ntraining), Ytr]

    margins = np.maximum(0, fscores - ftrue[:, np.newaxis] + delta)
    # margins = np.maximum(0, fscores - ftrue[:,] + delta)
    margins[np.arange(ntraining), Ytr] = 0.0

    loss = np.sum(margins) / ntraining
    loss += 0.5 * reg * np.sum(W_b * W_b)

    hold_margins = margins
    hold_margins[margins > 0] = 1

    row_sum = np.sum(hold_margins, axis=1)
    hold_margins[np.arange(ntraining), Ytr] = -1 * row_sum
    dW = Xtr_b.T.dot(hold_margins)
    dW /= ntraining * 1.0
    dW += reg * W_b

    return loss, dW


def add_bias_nodes_to_inputs(Xtr, Xval, Xtest):
    """ Add bias vector to the input mats"""

    Xtr_b = np.hstack((Xtr, np.ones((Xtr.shape[0], 1.0))))
    Xval_b = np.hstack((Xval, np.ones((Xval.shape[0], 1.0))))
    Xtest_b = np.hstack((Xtest, np.ones((Xtest.shape[0], 1.0))))

    return Xtr_b, Xval_b, Xtest_b


def early_stop(arr, threshold):
    """ Monitor learning performance on validation set.
    """
    count = np.count_nonzero(np.diff(arr) <= threshold)
    if count == (len(arr)-1):
        return True
    else:
        return False


def get_accuracy(X, W, Y):
    """ Extract accuracy values for given actual values output vector.
    """
    scores = X.dot(W)
    predictions = np.argmax(scores, axis=1)
    acc = np.mean(predictions == Y) * 100.0

    return acc, predictions


def construct_conf_mat(Yorig, Ypred, nobjs):
    """ Create the confusion matrix by using actual id and predicted ids.
    """

    csize = nobjs + 1 # eliminate zero as class
    conf_mat = np.zeros((csize, csize))
    for i in range(len(Yorig)):
        conf_mat[Yorig[i], Ypred[i]] += 1

    return conf_mat


def extract_inds_of_higest_acc(modality, header):
    """ Extract the higest accuracy based of CV-parameters, stepsize and regularization strength,
        return indices of the best iteration.
    """
    lstepsize, lreg= 7, 7,
    inds, accs = [], []
    for i in range(lstepsize):
        for ii in range(lreg):
            yactpred = 'results1/' + modality+'/yact_pred_' + str(i) + '_' + str(ii) + '.mat'
            matcont = prep.extract_mat_content(yactpred, header)
            yactual, ypred = matcont[0], matcont[1]
            accuracy = np.mean(yactual == ypred)

            accs.append(accuracy)
            inds.append([i, ii])

    return inds[np.argmax(accs)][0], inds[np.argmax(accs)][1], max(accs)


def report_mmodal_accs(modality, header):
    """ Display best accuracies with indices based on modalities"""
    i_rgb, ii_rgb, acc_rgb = extract_inds_of_higest_acc(modality[0], header)
    i_mask, ii_mask, acc_mask = extract_inds_of_higest_acc(modality[1], header)
    i_efus, ii_efus, acc_efus = extract_inds_of_higest_acc(modality[2], header)
    i_lfus, ii_lfus, acc_lfus = extract_inds_of_higest_acc(modality[3], header)

    print "Modality, indices  accuracy"
    print "============================="
    print "RGB:    ", i_rgb, ii_rgb, acc_rgb
    print "Depth:  ", i_mask, ii_mask, acc_mask
    print "Efus:   ", i_efus, ii_efus, acc_efus
    print "Lfus:   ", i_lfus, ii_lfus, acc_lfus
