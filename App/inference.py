import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from utils import get_final_preds
from HRnet import HRNet
from ResNet152 import ResNet152

class SimpleHRNet:
    def __init__(self,
                 c,
                 key,
                 checkpoint_path,
                 model_name='HRNet',
                 resolution=(256, 192),
                 interpolation=cv2.INTER_CUBIC,
                 multiperson=True,
                 return_heatmaps=False,
                 return_bounding_boxes=False,
                 max_batch_size=32,
                 device=torch.device('cpu')):

        self.c = c
        self.key = key
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.resolution = resolution  # in the form (height, width) as in the original implementation
        self.interpolation = interpolation
        self.multiperson = multiperson
        self.return_heatmaps = return_heatmaps
        self.return_bounding_boxes = return_bounding_boxes
        self.max_batch_size = max_batch_size
        self.device = device

        if model_name in ('HRNet', 'hrnet'):
            self.model = HRNet(c=c, key=key)
        else:
            raise ValueError('Wrong model name.')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        if 'cuda' in str(self.device):
            print("device: 'cuda' - ", end="")

            if 'cuda' == str(self.device):
                # if device is set to 'cuda', all available GPUs will be used
                print("%d GPU(s) will be used" % torch.cuda.device_count())
                device_ids = None
            else:
                # if device is set to 'cuda:IDS', only that/those device(s) will be used
                print("GPU(s) '%s' will be used" % str(self.device))
                device_ids = [int(x) for x in str(self.device)[5:].split(',')]

        elif 'cpu' == str(self.device):
            print("device: 'cpu'")
        else:
            raise ValueError('Wrong device name.')

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
                transforms.ToTensor()
        ])

    # Image is passed from cv2
    def predict_single(self, image):
        old_res = image.shape
        scale = (old_res[1], old_res[0])

        # Transformation   

        image = cv2.resize(
            image,
            (self.resolution[1], self.resolution[0]),
        interpolation=self.interpolation
        )

        image = self.transform(image).to(self.device)

        if len(image.shape) == 3:
            image = image.unsqueeze(dim=0)

        # Model Prediction
        self.model.eval()
        out = self.model(image)

        preds, maxvals = get_final_preds(out, scale)

        return preds, maxvals


class SimpleResNet152:
    def __init__(
        self,
        num_class,
        checkpoint_path,
        resolution=(224,224),
        device=torch.device('cpu')):

        self.num_class = num_class
        self.checkpoint_path = checkpoint_path
        self.resolution = resolution
        self.device = device
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.model = ResNet152(num_classes=self.num_class)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        if 'cuda' in str(self.device):
            print("device: 'cuda' - ", end="")

            if 'cuda' == str(self.device):
                # if device is set to 'cuda', all available GPUs will be used
                print("%d GPU(s) will be used" % torch.cuda.device_count())
                device_ids = None
            else:
                # if device is set to 'cuda:IDS', only that/those device(s) will be used
                print("GPU(s) '%s' will be used" % str(self.device))
                device_ids = [int(x) for x in str(self.device)[5:].split(',')]

        elif 'cpu' == str(self.device):
            print("device: 'cpu'")
        else:
            raise ValueError('Wrong device name.')

        self.model = self.model.to(self.device)

        self.transform_single = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.resolution[0], self.resolution[1]))
        ])

        self.transform_batch = transforms.Resize((self.resolution[0], self.resolution[1]))

    def predict_single(self, image):
        
        image = self.transform_single(image).to(self.device)
        
        assert len(image.shape) == 3, 'It is not a single image since length of image\'s shape is not equal to 3'

        image = image.unsqueeze(dim=0)
        self.model.eval()
        output = self.model(image).softmax(dim=1)
        idx = torch.argmax(output, dim=1)
        confidence = output[0][idx].item()
        certainty = (confidence - (1/self.num_class)) / (1/self.num_class)
        
        return idx, certainty

    def predict_batch(self, images):

        images = self.transform_batch(images).to(self.device)

        assert len(images.shape) == 4, 'It is not a single image since length of image\'s shape is not equal to 4'

        self.model.eval()
        output = self.model(images).softmax(dim=1)
        confidence, idx = torch.max(output, dim=1)
        certainty = (confidence - (1/self.num_class)) / (1/self.num_class)
        
        return idx, certainty
        

if __name__ == '__main__':
    import torch
    from torch import nn
    import cv2
    import os

    # simplehrnet = SimpleHRNet(c=48, key=12, checkpoint_path='./logs/20221220_1651/checkpoint_best_loss_0.0010082135344610403.pth')
    # Instruction = True

    # for image in os.listdir('./datasets/COCO/default'):
    #     if Instruction == True:
    #         image = cv2.imread(os.path.join('./datasets/COCO/default', image))
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         out = simplehrnet.predict_single(image)
            
    #         # Out: Batch, Points, XY

    #         for pair in out[0][0]:
    #             image = cv2.circle(image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
        
    #         cv2.imshow('Image', cv2.resize(image, (1000, 1000)))
    #         cv2.waitKey(1)

    trans = transforms.Compose([
        transforms.Resize((224,224))
    ])

    images = np.random.random((6, 175, 175, 3))
    images = np.transpose(images, (0, 3, 1, 2))
    
    images = torch.tensor(images, dtype=torch.float32)
    trans = transforms.Resize((224, 224))
    images = trans(images)
    print(images)
    



    model = SimpleResNet152(num_class=2, checkpoint_path='logs/020223_104428/checkpoint_best_acc_0.8055555555555556.pth')
    output = model.predict_batch(images)
    print(output.shape)