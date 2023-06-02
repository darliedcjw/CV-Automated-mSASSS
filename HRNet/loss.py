import torch.nn as nn
from misc.utils import compute_distance

# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True, use_point_dist=True):
        """
        MSE loss between output and GT points
        Args:
            use_target_weight (bool): use target weight.
                WARNING! This should be always true, otherwise the loss will sum the error for non-visible joints too.
        """
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.use_point_dist = use_point_dist

    def forward(self, output, target, target_distance, points_vis, points_scale, device, target_weight=None):
        # Include Points_dis
        # Output: Batch, Points, Y, X
        batch_size = output.shape[0]
        num_points = output.shape[1]
        heatmaps_pred = output.reshape((batch_size, num_points, -1)).split(1, 1) # Points, Batch, 1, -1
        heatmaps_gt = target.reshape((batch_size, num_points, -1)).split(1, 1) # Points, Batch, 1, -1
        loss = 0

        if self.use_point_dist and self.use_target_weight:   
            if target_weight is None:
                raise NameError

            output_distance = compute_distance(output, points_vis, points_scale).to(device) # Work within compute distance

            for idx in range(num_points):
                heatmap_pred = heatmaps_pred[idx].squeeze() # Flatten last 2 dimensions
                heatmap_gt = heatmaps_gt[idx].squeeze() # Flatten last 2 dimensions

                loss += 1 * 0.5 * (self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]), 
                    heatmap_gt.mul(target_weight[:, idx])
                    ))
            
            loss += 0.1 * 0.5 * (self.criterion(output_distance, target_distance))        

        elif self.use_target_weight and not self.use_point_dist:     
            if target_weight is None:
                raise NameError

            for idx in range(num_points):
                heatmap_pred = heatmaps_pred[idx].squeeze() # Flatten last 2 dimensions
                heatmap_gt = heatmaps_gt[idx].squeeze() # Flatten last 2 dimensions

                loss += 0.5 * (self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
        
        elif self.use_point_dist and not self.use_target_weight:
            loss += 0.5 * (self.criterion(heatmap_pred, heatmap_gt) + self.criterion(output_distance, target_distance))

        else:
            loss += 0.5 * (self.criterion(heatmap_pred, heatmap_gt))

        return loss / num_points