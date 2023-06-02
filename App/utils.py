import numpy as np
import cv2
import torch
import math
import kornia as K

'''
Transformation
scale: width, height
'''
def get_resize_AF(original_size, output_size, train=False):

    if train:
        src = torch.zeros((4,2), dtype=torch.float32)
        dst = torch.zeros((4,2), dtype=torch.float32)

        src[1, :] = [original_size[0], 0]
        src[2, :] = [0, original_size[1]]
        src[3, :] = [original_size[0], original_size[1]]

        dst[1, :] = [output_size[0], 0]
        dst[2, :] = [0, output_size[1]]
        dst[3, :] = [output_size[0], output_size[1]]

        return K.geometry.get_perspective_transform(src, dst)

    else:
        src = np.zeros((3,2), dtype=np.float32)
        dst = np.zeros((3,2), dtype=np.float32)

        src[1, :] = [original_size[0], 0]
        src[2, :] = [0, original_size[1]]

        dst[1, :] = [output_size[0], 0]
        dst[2, :] = [0, output_size[1]]

        return cv2.getAffineTransform(src, dst)


def affine_transform(pt, t, train=False):
    '''
    pt has x, y coordinates and t is the transformation matrix
    '''

    if train:
        new_pt = torch.tensor([pt[0], pt[1], 1.].T)
        new_pt = torch.dot(t, new_pt)

    else:
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
    
    return new_pt[:2]


def get_angle(pt1, pt2):
    '''
    Use dot product
    '''
    try:
        y_len = pt2[1] - pt1[1]
        x_len = pt2[0] - pt1[0]
        rot_deg = math.degrees(math.atan(y_len/x_len))
    except ValueError:
        rot_deg = 0

    return rot_deg

'''
Evaluation
'''
def evaluate_pck_accuracy(output, target, hm_type='gaussian', thr=0.02):
    """
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than y,x locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    """
    idx = list(range(output.shape[1]))
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output) # (Batch, 17, 2) where 2: Column and Row Number (Confined within 64, 48)
        target, _ = get_max_preds(target) # Column and Row Number (Confined within 64, 48)
        h = output.shape[2]
        w = output.shape[3]
        norm = torch.ones((pred.shape[0], 2)) * torch.tensor([w, h], dtype=torch.float32)
        norm = norm.to(output.device)
    else:
        raise NotImplementedError
    dists = calc_dists(pred, target, norm) # (17 by Batch) where value = Norm

    acc = torch.zeros(len(idx)).to(dists.device)
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i] = dist_acc(dists[idx[i]], thr=thr)
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else torch.tensor(0)
    return acc, avg_acc, cnt, pred, target


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps (Batch, 17, 2) where 2: Width, Height
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    # assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2) # Output: (Batch, 17)

    maxvals = maxvals.unsqueeze(dim=-1) # Add 1 more dimension at the end
    idx = idx.float()

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps.device)

    preds[:, :, 0] = idx % width  # X
    preds[:, :, 1] = torch.floor(idx / width)  # Y, by rounding it to least whole number

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device) # Greater than 0 return True

    preds *= pred_mask
    return preds, maxvals


def calc_dists(preds, target, normalize):
    preds = preds.type(torch.float32)
    target = target.type(torch.float32)
    dists = torch.zeros((preds.shape[1], preds.shape[0])).to(preds.device)
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1: # Check that target has visible points
                
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
            
                dists[c, n] = torch.linalg.norm(normed_preds - normed_targets) # Square of Sum of Square

                # dists[c, n]
            else:
                dists[c, n] = -1
    return dists
 

def dist_acc(dists, thr=0.03):
    """
    1 Pixel difference in x, 1 pixel difference in y result in threshold 0.026
    Return percentage below threshold while ignoring values with a -1
    dist: Scores of Joint Idx, c
    """
    dist_cal = torch.ne(dists, -1) # ne: Not Equal (Boolean)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return torch.lt(dists[dist_cal], thr).float().sum() / num_dist_cal
    else:
        return -1
    

'''
Inference Methods
'''
def get_final_preds(batch_heatmaps, scale, train=False):
    '''
    Scale: Original_Width, Orignal Height
    '''
    coords, maxvals = get_max_preds(batch_heatmaps) # Coords: Batch, 12, 2 [Tensors]

    h = batch_heatmaps.shape[2]
    w = batch_heatmaps.shape[3]

    if train:   
        for b in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                p_hm = batch_heatmaps[b][p].detach()
                px = int(torch.floor(coords[b][p][0] + 0.5))
                py = int(torch.floor(coords[b][p][1] + 0.5))
                if 1 < px < (w - 1) and 1 < py < (h - 1):
                    diff = torch.tensor(
                            [p_hm[py][px + 1] - p_hm[py][px - 1], 
                            (p_hm[py + 1][px] - p_hm[py - 1][px])
                            ]
                        ).to('cuda')
                    
                    coords[b][p] += torch.sign(diff) * .25

        #Single batch: scale is 2 dimensional
        if coords.shape[0] == 1:
            for b in range(coords.shape[0]):
                coords[b] = transform_preds(
                    coords=coords[b],
                    original_size=scale[b],
                    output_size=[w, h],
                    train=train
                )

    else:
        maxvals = maxvals.detach().cpu().numpy()
        coords = coords.detach().cpu().numpy()

        #Single inference: scale is 1 dimensional
        if coords.shape[0] == 1:
            for b in range(coords.shape[0]):
                coords[b] = transform_preds(
                    coords=coords[b],
                    original_size=scale,
                    output_size=[w, h]
        )

        #Batch inference: scale is 2 dimensional
        elif coords.shape[0] > 1:
            for b in range(coords.shape[0]):
                coords[b] = transform_preds(
                    coords=coords[b],
                    original_size=scale[b],
                    output_size=[w, h]
        )
    
    return coords, maxvals


def transform_preds(coords, original_size, output_size, train=False):
    '''
    Transformation to original size
    '''
    if train:
        target_coords = torch.zeros(coords.shape)
        trans = get_resize_AF(original_size=output_size, output_size=original_size)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = affine_transform(
                    coords[p, 0:2],
                    trans,
                    train=train
                    )

    else:
        target_coords = np.zeros(coords.shape)
        trans = get_resize_AF(original_size=output_size, output_size=original_size)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = affine_transform(
                    coords[p, 0:2],
                    trans
                    )

    return target_coords


'''
Training
'''
def compute_distance(output, points_vis, points_scale):
    batch_size = output.shape[0]
    num_points = output.shape[1]

    preds, _ = get_final_preds(output, points_scale, train=True) # Include False and True for train

    preds_dist = torch.zeros((batch_size, num_points // 2, 2), dtype=torch.float32)

    for b in range(batch_size):
        for pt in range(num_points):
            if pt < (num_points - 2) and pt % 2 == 0: # Less than 10 and 0,2,4,6,8
                if points_vis[b ,pt, 0] > 0 and points_vis[b, (pt + 2), 0] > 0: #####

                    # Right/Left
                    x0_dist = abs(preds[b, pt, 0] - preds[b, (pt + 1), 0]) / points_scale[b][0]
                    y0_dist = abs(preds[b, pt, 1] - preds[b, (pt + 1), 1]) / points_scale[b][1]
                    c0 = torch.tensor([x0_dist, y0_dist], dtype=torch.float32)
                    preds_dist[b, pt // 2, 0] = torch.linalg.norm(c0)

                    # Down
                    x1_dist = abs(preds[b, pt, 0] - preds[b, (pt + 2), 0]) / points_scale[b][0]
                    y1_dist = abs(preds[b, pt, 1] - preds[b, (pt + 2), 1]) / points_scale[b][1]
                    c1 = torch.tensor([x1_dist, y1_dist], dtype=torch.float32)
                    preds_dist[b, pt // 2, 1] = torch.linalg.norm(c1)

                elif points_vis[b, pt, 0] > 0 and points_vis[b, (pt + 2), 0] == 0:

                # Right/Left 
                    x0_dist = abs(preds[b, pt, 0] - preds[b, (pt + 1), 0]) / points_scale[b][0]
                    y0_dist = abs(preds[b, pt, 1] - preds[b, (pt + 1), 1]) / points_scale[b][1]
                    c0 = torch.tensor([x0_dist, y0_dist], dtype=torch.float32)
                    preds_dist[b, pt // 2, 0] = torch.linalg.norm(c0)

                    # Down
                    preds_dist[b, pt // 2, 1] = 0
                
                else:
                    continue

            elif pt == (num_points - 2):
                if points_vis[b, pt, 0] > 0:
                    # Right/Left
                    x0_dist = abs(preds[b, pt, 0] - preds[b, (pt + 1), 0]) / points_scale[b][0]
                    y0_dist = abs(preds[b, pt, 1] - preds[b, (pt + 1), 1]) / points_scale[b][1]
                    c0 = torch.tensor([x0_dist, y0_dist], dtype=torch.float32)
                    preds_dist[b, pt // 2, 0] = torch.linalg.norm(c0)

                    # Down
                    preds_dist[b, pt // 2, 1] = 0
                else:
                    continue 

    return preds_dist


if __name__ == '__main__':
    test_heatmap = torch.randn(4, 12, 64, 48)
    scale = torch.randn(4,2)
    # scale = np.random.randn(4,2)
    preds, _ = get_final_preds(test_heatmap, scale, train=False)
    print(type(preds))
