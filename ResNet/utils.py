import os
import torch

torch.manual_seed(0)

def compute_accuracy(output, label, size):
    '''
    Compute accuracy
    output: (batch size x number of classes)
    label: batch size
    size: batch size
    '''
    pred = torch.argmax(output, dim=1)
    acc = torch.eq(pred, label).sum() / size
    
    return acc


def compute_precision(output, label, num_classes):
    '''
    Compute precision for each classes
    output: (batch size x number of classes)
    label: batch size
    num_classes = number of classes
    '''
    precision = torch.zeros(size=(num_classes,))
    precision_count = torch.zeros(size=(num_classes,))
    
    pred = torch.argmax(output, dim=1)

    for class_idx in range(num_classes):

        label_idx = (label == class_idx).nonzero().squeeze(dim=1)
        pred_idx = (pred == class_idx).nonzero().squeeze(dim=1)
        
        true_positive = 0
        
        total_pred_positive = (pred == class_idx).sum()

        for idx in label_idx:
            if idx in pred_idx:
                true_positive += 1
            else:
                continue
        
        precision[class_idx] = true_positive / total_pred_positive
    
    # Nan type handling
    precision_count += precision.isnan() == 0
    precision[precision.isnan()] = 0
    
    return precision, precision_count


def compute_recall(output, label, num_classes):
    '''
    Compute precision for each classes
    output: (batch size x number of classes)
    label: batch size
    num_classes = number of classes
    '''
    recall = torch.zeros(size=(num_classes,))
    recall_count = torch.zeros(size=(num_classes,))
    
    pred = torch.argmax(output, dim=1)

    for class_idx in range(num_classes):

        label_idx = (label == class_idx).nonzero().squeeze(dim=1)
        pred_idx = (pred == class_idx).nonzero().squeeze(dim=1)
        
        true_positive = 0
        
        total_gt_positive = (label == class_idx).sum()

        for idx in label_idx:
            if idx in pred_idx:
                true_positive += 1
            else:
                continue
        
        recall[class_idx] = true_positive / total_gt_positive

    recall_count += recall.isnan() == 0
    recall[recall.isnan()] = 0
    
    return recall, recall_count


def compute_f1_score(precision, recall):
   '''
   Compute f1 score for each class
   precision: number of classes
   recall: number of classes
   '''
   f1 = 2 * (precision * recall) / (precision + recall)
   
   f1[f1.isnan()] = 0
   
   return f1

def save_checkpoint(path, epoch, model, optimizer, params=None):
    """
    Save a PyTorch checkpoint.
    Args:
        path (str): path where the checkpoint will be saved.
        epoch (int): current epoch.
        model (torch.nn.Module): model whose parameters will be saved.
        optimizer (torch.optim.Optimizer): optimizer whose parameters will be saved.
        params (dict): other parameters. Optional.
            Default: None
    """
    if os.path.isdir(path):
        path = os.path.join(path, 'checkpoint.pth')
    torch.save(
        {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        },
        path
    )


def load_checkpoint(path, model, optimizer=None, device=None):
    """
    Load a PyTorch checkpoint.
    Args:
        path (str): checkpoint path.
        model (torch.nn.Module): model whose parameters will be loaded.
        optimizer (torch.optim.Optimizer): optimizer whose parameters will be loaded. Optional.
            Default: None
        device (torch.device): device to be used.
            Default: None
    Returns:
        epoch (int): saved epoch
        model (torch.nn.Module): reference to `model`
        optimizer (torch.nn.Optimizer): reference to `optimizer`
        params (dict): other saved params
    """
    if os.path.isdir(path):
        path = os.path.join(path, 'checkpoint.pth')
    checkpoint = torch.load(path, map_location=device)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    params = checkpoint['params']

    return epoch, model, optimizer, params