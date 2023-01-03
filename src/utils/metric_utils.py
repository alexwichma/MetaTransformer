"""
Contains utility functions and classes around metric tracking and metric calculations
"""


from __future__ import absolute_import, division, print_function

import pandas as pd
import torch


def calc_precision(tp, classified):
    if classified == 0:
        return 0
    return tp / classified


def calc_recall(tp, total):
    if total == 0:
        return 0
    return tp / total


def calc_f1(prec, rec):
    if (prec + rec) == 0:
        return 0
    return 2 * prec * rec / (prec + rec)


def average_n_predictions(predictions, num_classes, n):
    # dim(predictions) should be (num_predictions, num_classes)
    # reshape to (-1, predictions per element, num_classes)
    # average along predictions per element dimension (=> dim 1)
    return torch.mean(predictions.view(-1, n, num_classes), dim=1)


def read_level_metrics(predictions, labels, threshold):
    prec = precision(predictions, labels, threshold)
    rec = recall(predictions, labels, threshold)
    f1 = calc_f1(prec, rec)
    return prec, rec, f1


def precision(predictions, labels, threshold=.5):
    vals_max, args_max = torch.max(predictions, 1)
    classified = vals_max.ge(threshold)
    num_classified = classified.sum().item()
    base_correct = args_max == labels
    num_correctly_classified = torch.logical_and(classified, base_correct).sum().item()
    return calc_precision(num_correctly_classified, num_classified)


def recall(predictions, labels, threshold=.5):
    num_total_predictions = labels.shape[0]
    vals_max, args_max = torch.max(predictions, 1)
    classified = vals_max.ge(threshold)
    base_correct = args_max == labels
    num_correctly_classified = torch.logical_and(classified, base_correct).sum().item()
    return calc_recall(num_correctly_classified, num_total_predictions)


def multi_lv_precision_factory(level_idx, rank_name, threshold):
    
    def fn1(predictions, labels):
        lv_predictions = predictions[level_idx]
        lv_labels = labels[level_idx]
        return precision(lv_predictions, lv_labels, threshold)
    
    fn1.__name__ = f"precision_{rank_name}_{level_idx}"
    return fn1


def multi_lv_recall_factory(level_idx, rank_name, threshold):
    
    def fn1(predictions, labels):
        lv_predictions = predictions[level_idx]
        lv_labels = labels[level_idx]
        return recall(lv_predictions, lv_labels, threshold)
    
    fn1.__name__ = f"recall_{rank_name}_{level_idx}"
    return fn1


def community_level_metrics(predictions, labels, num_classes, threshold=.5, cut_off=0.0):
    """ Community-level metric calculation

    Based on the predictions, the ground-truth labels, a classification threshold and a abundance cut-off
    this function calculates community level precision and recall. Additionally, it calculcates the normalized
    per-class abundance predictions and the absolute error between ground-truth and predicted abundance.
    """
    pred_val_max, pred_arg_max = torch.max(predictions, dim=1)
    # filter out predictions that does not meet threshold
    valid_preds_idx = pred_val_max >= threshold
    valid_preds = pred_arg_max[valid_preds_idx]
    valid_labels = labels[valid_preds_idx]
    
    # class abundance calculcation => get counts for each class and normalize them
    unique_pred_elems, unique_pred_counts = torch.unique(valid_preds, return_counts=True)
    norm_counts = unique_pred_counts.float() / torch.sum(unique_pred_counts).float()  # normalize the counts
    # abundances as predicted by the model
    predicted_abundances = torch.zeros((num_classes,), dtype=torch.float32)
    predicted_abundances[unique_pred_elems] = norm_counts * 100.0
    # ground truth abundances extracted from the labels
    unique_gt_labels, unique_gt_counts = torch.unique(labels, return_counts=True)
    norm_gt_counts = unique_gt_counts / torch.sum(unique_gt_counts).float() * 100.0
    gt_abundances = torch.zeros((num_classes,), dtype=torch.float32)
    gt_abundances[unique_gt_labels] = norm_gt_counts
    # we also calculate the abundance error here
    abundance_errors = predicted_abundances - gt_abundances

    # find classes that reach abundance cut-off
    # this needs to be done to make sure that all classes actually exist (some classes can be never predicted)
    class_abundance_ok = torch.zeros((num_classes,), dtype=torch.bool)
    class_cutoff_reached_idx = norm_gt_counts >= cut_off
    class_abundance_ok[unique_gt_labels[class_cutoff_reached_idx]] = True
    # get a boolean mask for the valid predictions, indicating which prediction reached the abundance cut-off
    preds_cutoff_reached_mask = class_abundance_ok[valid_preds]

    taxa_idf = valid_preds[preds_cutoff_reached_mask]
    taxa_idf_labels = valid_labels[preds_cutoff_reached_mask]
    
    num_taxa_idf = taxa_idf.shape[0]
    taxa_corr_idf = taxa_idf == taxa_idf_labels
    num_tax_corr_idf = taxa_corr_idf.sum().item()
 
    # count number of unique correct class indices
    num_unique_corr_cls = torch.unique(taxa_idf[taxa_corr_idf]).shape[0]
    # count number of unique class indices in truth set
    num_elems_in_truth_set = class_abundance_ok.sum().item()

    prec = num_tax_corr_idf / num_taxa_idf
    rec = num_unique_corr_cls / num_elems_in_truth_set

    return prec, rec, [predicted_abundances.numpy(), gt_abundances.numpy(), abundance_errors.numpy()]


def get_abundance(predictions, num_classes, threshold=.5):
    """ Abundance calculation

    Based on the predictions, the ground-truth labels, a classification threshold and a abundance cut-off
    this function calculates the normalized per-class abundance predictions and the total abundance.
    """

    pred_val_max, pred_arg_max = torch.max(predictions, dim=1)
    # filter out predictions that does not meet threshold
    valid_preds_idx = pred_val_max >= threshold
    valid_preds = pred_arg_max[valid_preds_idx]
    
    # class abundance calculcation => get counts for each class and normalize them
    unique_pred_elems, unique_pred_counts = torch.unique(valid_preds, return_counts=True)
    abundance = torch.zeros((num_classes,), dtype=torch.long)
    abundance[unique_pred_elems] = unique_pred_counts 
    norm_counts = unique_pred_counts.float() / torch.sum(unique_pred_counts).float()  # normalize the counts
    # abundances as predicted by the model
    predicted_abundances = torch.zeros((num_classes,), dtype=torch.float32)
    predicted_abundances[unique_pred_elems] = norm_counts * 100.0
    
    return abundance.numpy(), predicted_abundances.numpy()


class MetricTracker:
    """ Tracks metrics

    Tracks metrics under the specified keys. It allows to add to a metric and get its average
    """

    def __init__(self, *keys, postfix='') -> None:
        self.postfix = postfix
        if self.postfix:
            keys = [k + postfix for k in keys]
        self.metrics = pd.DataFrame(columns=['total', 'counts', 'average'], index=keys)
        self.reset_metrics()

    def reset_metrics(self):
        for col in self.metrics.columns:
            self.metrics[col].values[:] = 0

    def add_metric(self, key, value):
        if self.postfix:
            key = key + self.postfix
        self.metrics.total[key] += value
        self.metrics.counts[key] += 1
        self.metrics.average[key] = self.metrics.total[key] / self.metrics.counts[key]

    def get_avg(self, key):
        if self.postfix:
            key = key + self.postfix
        return self.metrics.average[key]

    def get_avgs(self):
        return dict(self.metrics.average)
