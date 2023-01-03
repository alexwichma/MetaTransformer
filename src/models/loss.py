import torch


def cross_entropy_loss():
    return torch.nn.CrossEntropyLoss()


def multi_level_cross_entropy_loss(class_weights, ignore_index=-1):
    """Cross-entropy loss adopted for multi-level classification"""
    # class weights => list of weight tensor for each level prediction level
    # predictions => list of softmaxed predictions
    # labels => tensor of labels for each element in batch
    def evaluate_multi_level_loss(predictions, labels):
        complete_loss = None
        for index, lv_preds in enumerate(predictions):
            lv_labels = labels[index]
            lv_class_weights = class_weights[index] if class_weights else None
            ce_loss = torch.nn.CrossEntropyLoss(weight=lv_class_weights, ignore_index=ignore_index)
            loss = ce_loss(lv_preds, lv_labels)
            if complete_loss is None:
                complete_loss = loss
            else:
                complete_loss += loss
        return complete_loss
    
    return evaluate_multi_level_loss
