import os
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torchvision import transforms
from pycocotools.coco import COCO
from misc.utils import get_resize_AF, affine_transform, evaluate_pck_accuracy

class COCODataset():
    def __init__(self,
    root_path='./datasets/COCO',
    data_version='default',
    keys=12,
    use_dist=True,
    is_train=True,
    is_flip=True,
    is_scale=True,
    is_rotate=True,
    image_height=256,
    image_width=192,
    color_rgb=True,
    scale_prob=0.5,
    scale_factor=0.1,
    flip_prob=0.5,
    rotate_prob=0.5,
    rotation_factor=10.,
    heatmap_sigma=3, # Changable
    ):
        self.root_path = root_path
        self.data_version = data_version
        self.is_train = is_train
        self.is_flip = is_flip
        self.is_scale = is_scale
        self.is_rotate = is_rotate
        self.image_width = image_width
        self.image_height = image_height
        self.color_rgb = color_rgb
        self.scale_prob = scale_prob
        self.scale_factor = scale_factor
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotation_factor = rotation_factor
        self.heatmap_sigma = heatmap_sigma

        self.data_path = os.path.join(self.root_path, self.data_version)
        self.data_path = os.path.join(self.root_path, self.data_version)
        
        self.annotation_path = os.path.join(
            self.root_path, 'annotations', 'person_keypoints_{}.json'.format(self.data_version)
        )

        self.image_size = (self.image_width, self.image_height)
        self.heatmap_size = (int(self.image_width/4), int(self.image_height/4)) # Final Output 48 x 68
        self.heatmap_type = 'gaussian'
        self.keys = keys
        self.use_dist = use_dist

        self.transform = transforms.Compose([
            transforms.ToTensor()
            ])

        self.coco = COCO(self.annotation_path)

        # Get the list of Image Ids
        self.imgIds = self.coco.getImgIds()       
        
        self.data = []

        for imgId in tqdm(self.imgIds):
            ann_ids = self.coco.getAnnIds(imgIds=imgId, iscrowd=False)
            img = self.coco.loadImgs(imgId)[0]

            # Load all annotations
            objs = self.coco.loadAnns(ann_ids)
            
            # To account for images that has not been annotated (~100)
            if len(objs) == 0:
                continue

            else:         
                valid_objs = []

                for obj in objs:
                    x1 = 0
                    y1 = 0
                    x2 = img['width'] - 1
                    y2 = img['height'] - 1

                    # Use only valid bounding boxes
                    if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                        obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                        valid_objs.append(obj)

                objs = valid_objs


                for obj in objs:
                    points = np.zeros((self.keys, 2), dtype=float) # 12x2
                    points_visibility = np.ones((self.keys, 2), dtype=float) # 12x2


                    for pt in range(self.keys):
                        if obj['keypoints'][pt * 3 + 0] == 0 and obj['keypoints'][pt * 3 + 1] == 0: # Annotation methods: Set visibility to 0
                            t_vis = 0
                            points_visibility[pt, 0] = t_vis
                            points_visibility[pt, 1] = t_vis


                        else:                     
                            points[pt, 0] = obj['keypoints'][pt * 3 + 0] # 1st Column: X-Axis
                            points[pt, 1] = obj['keypoints'][pt * 3 + 1] # 2nd Column: Y-Axis
                            t_vis = int(np.clip(obj['keypoints'][pt * 3 + 2], 0, 1)) # Clipping it between 0 & 1
                            points_visibility[pt, 0] = t_vis
                            points_visibility[pt, 1] = t_vis

                center, scale = self._box2center(obj['clean_bbox'][:4])

                self.data.append({
                    'imgId': imgId,
                    'annId': obj['id'],
                    'imgPath': os.path.join(self.root_path, self.data_version, img['file_name']),
                    'center': center,
                    'scale': scale, # Width, Height
                    'points': points,
                    'points_visibility': points_visibility,
                })


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        points_data = self.data[index].copy()
        '''
        IMREAD_COLOR: Only load images with color.
        IMREAD_IGNORE_ORIENTATION: Ignore rotation of camera
        Height x Width x Channel
        '''
        image = cv2.imread(points_data['imgPath'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        if self.color_rgb:            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            raise ValueError('Fail to read {}'.format(image))

        points = points_data['points']
        points_vis = points_data['points_visibility']  
        center = points_data['center']
        scale = points_data['scale']
        
        s = 1
        r = 0
        if self.is_train:
            rf = self.rotation_factor
            sf = self.scale_factor
        
            if self.is_scale:
                if torch.rand((1)) < self.scale_prob:
                    s = s * np.clip(torch.randn((1)).item() * sf + 1, 1 - sf, 1 + sf)
            
            if self.is_rotate:
                # Rotate Positive is anti-clockwise
                if torch.rand((1)) < self.rotate_prob:
                    r = np.clip(torch.randn((1)).item() * rf, -rf, rf) # Clip between (-rf*2, rf*2)

            if self.is_flip:
                if torch.rand((1)) < self.flip_prob:
                    image = image[:, ::-1, :] # Rows x Columns and only Columns is flipped
                    points[:, 0] = image.shape[1] - points[:, 0] - 1
                    center[0] = image.shape[1] - center[0] - 1

        affine_transformation = [
            get_resize_AF(scale, self.image_size),
            cv2.getRotationMatrix2D(np.array(self.image_size, dtype=np.float32) // 2 - 1, r, s)
        ]

        for trans in affine_transformation:
            image = cv2.warpAffine(
                image,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR
            )

            for i in range(self.keys):
                if points_vis[i, 0] > 0.:
                    points[i, 0:2] = affine_transform(points[i, 0:2], trans)

        if self.transform is not None:
            image = self.transform(image)

        target, target_weight = self._generate_target(points, points_vis)


        '''
        Distance between points for computing loss
        '''
        points_distance = np.zeros((self.keys // 2, 2), dtype=np.float32)
        
        if self.use_dist:
            for pt in range(self.keys):
                if pt < (self.keys - 2) and pt % 2 == 0: # Less than 10 and 0,2,4,6,8
                    if points_vis[pt, 0] > 0 and points_vis[(pt + 2), 0] > 0:

                        # Right/Left
                        x0_dist = abs(points[pt, 0] - points[(pt + 1), 0]) / scale[0]
                        y0_dist = abs(points[pt, 1] - points[(pt + 1), 1]) / scale[1]
                        c0 = np.array([x0_dist, y0_dist], dtype=np.float32)
                        points_distance[pt // 2, 0] = np.linalg.norm(c0)

                        # Down
                        x1_dist = abs(points[pt, 0] - points[(pt + 2), 0]) / scale[0]
                        y1_dist = abs(points[pt, 1] - points[(pt + 2), 1]) / scale[1]
                        c1 = np.array([x1_dist, y1_dist], dtype=float)
                        points_distance[pt // 2, 1] = np.linalg.norm(c1)

                    elif points_vis[pt, 0] > 0 and points_vis[(pt + 2), 0] == 0:

                    # Right/Left 
                        x0_dist = abs(points[pt, 0] - points[(pt + 1), 0]) / scale[0]
                        y0_dist = abs(points[pt, 1] - points[(pt + 1), 1]) / scale[1]
                        c0 = np.array([x0_dist, y0_dist], dtype=np.float32)
                        points_distance[pt // 2, 0] = np.linalg.norm(c0)

                        # Down
                        points_distance[pt // 2, 1] = 0
                    
                    else:
                        continue

                elif pt == (self.keys - 2):
                    if points_vis[pt, 0] > 0:
                        # Right/Left
                        x0_dist = abs(points[pt, 0] - points[(pt + 1), 0]) / scale[0]
                        y0_dist = abs(points[pt, 1] - points[(pt + 1), 1]) / scale[1]
                        c0 = np.array([x0_dist, y0_dist], dtype=np.float32)
                        points_distance[pt // 2, 0] = np.linalg.norm(c0)

                        # Down
                        points_distance[pt // 2, 1] = 0
                    else:
                        continue
            else:
                pass


        # Update metadata
        points_data['points'] = points
        points_data['points_visibility'] = points_vis
        points_data['points_distance'] = points_distance
        points_data['center'] = center
        points_data['scale_factor'] = s
        points_data['rotation_factor'] = r

        return image, target.astype(np.float32), target_weight.astype(np.float32), points_data

    
    def evaluate_accuracy(self, output, target, params=None):
        if params is not None:
            hm_type = params['hm_type']
            thr = params['thr']
            accs, avg_acc, cnt, joints_preds, joints_target = evaluate_pck_accuracy(output, target, hm_type, thr)
        else:
            accs, avg_acc, cnt, joints_preds, joints_target = evaluate_pck_accuracy(output, target, hm_type='gaussian', thr=0.03)

        return accs, avg_acc, cnt, joints_preds, joints_target


    """Private Methods"""

    def _box2center(self, box):
        x, y, w, h = box[:4]
        center = np.zeros((2,), dtype=np.float32)
        scale = np.zeros((2,), dtype=np.float32)

        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        scale[0] = w
        scale[1] = h        
        
        return center, scale

    def _generate_target(self, points, points_vis):
        """
        :param points:  [keys, 2]
        :param points_vis: [keys, 2]
        :return: target, target_weight(1: visible, 0: invisible)
        """
        target_weight = np.ones((self.keys, 1), dtype=np.float32)
        target_weight[:, 0] = points_vis[:, 0] # Target_weight = Point Visibility

        # 17 x 48 x 68
        if self.heatmap_type == 'gaussian':
            target = np.zeros((self.keys,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=float)

            # self.heatmap_sigma = 3
            tmp_size = self.heatmap_sigma * 3

            for point_id in range(self.keys):
                feat_stride = np.asarray(self.image_size) / np.asarray(self.heatmap_size) # (image_width, image_height) / (heatmap_width, heatmap_height)
                
                # Each point coordinate scaled down to match heatmap size
                mu_x = int(points[point_id][0] / feat_stride[0] + 0.5) 
                mu_y = int(points[point_id][1] / feat_stride[1] + 0.5)
                
                # Check that any part of the gaussian is in-bounds
                # Upper Left Coordinates and Bottom Right Coordinates
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                
                # 48 x 68
                # If exceed or less than 0, then point visibility = 0
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[point_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis] # Increase dimension of column by 1
                x0 = y0 = size // 2
                
                # The gaussian is not normalized, we want the center value to equal 1
                # The heatmap_sigma is standard deviation
                # Gaussian Formula
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.heatmap_sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[point_id]
                if v > 0.5:
                    target[point_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        else:
            raise NotImplementedError

        return target, target_weight