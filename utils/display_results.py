import numpy as np
import sklearn.metrics as sk
# import figure
import sys
sys.path.append('/lustre/home/xmwang/codes/code_cc_misd/utils')
from save_items import append_to_csv_file



recall_level_default = 0.95
ood_save_dir = '/lustre/home/xmwang/codes/code_cc_misd/ood_result/ood_results_2.csv'

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    # print(examples.shape)
    # print(examples)
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    # print(labels)
    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def show_performance(pos, neg, in_data, ood_data, method_name='Ours', recall_level=recall_level_default, ood_save_dir=ood_save_dir):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)

    print_measures(auroc, aupr, fpr, in_data, ood_data, method_name, recall_level, ood_save_dir=ood_save_dir)
    # print('\t\t\t' + method_name)
    # print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    # print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    # print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
    # print('FDR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fdr))


def print_measures(auroc, aupr, fpr, in_data, ood_data, method_name='Ours', recall_level=recall_level_default, ood_save_dir= ood_save_dir):
    print('\t\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))
    dict_data = {'method_name': method_name,
                 'in_data': in_data,
                 'ood_dataset': ood_data,
                 'FPR95': '{:.2f}'.format(100 * fpr), 
                 'AUROC': '{:.2f}'.format(100 * auroc), 
                 'AUPR': '{:.2f}'.format(100 * aupr)}
    append_to_csv_file(dict_data, ood_save_dir)
    


def print_measures_with_std(aurocs, auprs, fprs, in_data, ood_data='mean', method_name='Ours', recall_level=recall_level_default, ood_save_dir=ood_save_dir):
    print('\t\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}\t+/- {:.2f}'.format(int(100 * recall_level), 100 * np.mean(fprs), 100 * np.std(fprs)))
    print('AUROC: \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(aurocs), 100 * np.std(aurocs)))
    print('AUPR:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(auprs), 100 * np.std(auprs)))
    dict_data = {'method_name': method_name,
                 'in_data': in_data,
                 'ood_dataset': ood_data,
                 'FPR95': '{:.2f}\t+/- {:.2f}'.format(100 * np.mean(fprs), 100 * np.std(fprs)),
                 'AUROC': '{:.2f}\t+/- {:.2f}'.format(100 * np.mean(aurocs), 100 * np.std(aurocs)), 
                 'AUPR': '{:.2f}\t+/- {:.2f}'.format(100 * np.mean(auprs), 100 * np.std(auprs))}
    append_to_csv_file(dict_data, ood_save_dir)
    


def show_performance_comparison(pos_base, neg_base, pos_ours, neg_ours, baseline_name='Baseline',
                                method_name='Ours', recall_level=recall_level_default):
    '''
    :param pos_base: 1's class, class to detect, outliers, or wrongly predicted
    example scores from the baseline
    :param neg_base: 0's class scores generated by the baseline
    '''
    auroc_base, aupr_base, fpr_base = get_measures(pos_base[:], neg_base[:], recall_level)
    auroc_ours, aupr_ours, fpr_ours = get_measures(pos_ours[:], neg_ours[:], recall_level)

    print('\t\t\t' + baseline_name + '\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}\t\t{:.2f}'.format(
        int(100 * recall_level), 100 * fpr_base, 100 * fpr_ours))
    print('AUROC:\t\t\t{:.2f}\t\t{:.2f}'.format(
        100 * auroc_base, 100 * auroc_ours))
    print('AUPR:\t\t\t{:.2f}\t\t{:.2f}'.format(
        100 * aupr_base, 100 * aupr_ours))
    # print('FDR{:d}:\t\t\t{:.2f}\t\t{:.2f}'.format(
    #     int(100 * recall_level), 100 * fdr_base, 100 * fdr_ours))
