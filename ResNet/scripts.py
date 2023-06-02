import torch

from torchvision import transforms as T
from torchvision.datasets import ImageFolder

import argparse

from train import Train

torch.manual_seed(0)

def main(
    train_path,
    val_path,
    log_path,
    num_classes,
    epochs,
    batch_size,
    learning_rate,
    lr_scheduler,
    momentum,
    optimizer,
    loss,
    num_workers,
    device,
    use_tensorboard
    ):

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        ])

    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')


    ds_train = ImageFolder(root=train_path, transform=transform)
    ds_val = ImageFolder(root=val_path, transform=transform)

    with open('class_idx.txt', 'w') as f:
        for idx in ds_train.class_to_idx:
            f.writelines('{}: {}\n'.format(idx, ds_train.class_to_idx[idx]))

    train = Train(
                ds_train, 
                ds_val,
                log_path,
                num_classes,
                epochs,
                batch_size,
                learning_rate,
                lr_scheduler,
                momentum,
                optimizer,
                loss,
                num_workers,
                device,
                use_tensorboard,
                )
    
    train.run()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', '-tp', help='Train folder', type=str, default='./datasets/1/train')
    parser.add_argument('--val_path', '-vp', help='Val folder', type=str, default='./datasets/1/val')
    parser.add_argument('--log_path', '-lp', help='Log folder', type=str, default='logs/SGD_1')
    parser.add_argument('--num_classes', '-c', help='Number of classes', type=int, default=2)
    parser.add_argument('--epochs', '-e', help='Number of epochs', type=int, default=300) 
    parser.add_argument('--batch_size', '-b', help='Training batch size', type=int, default=8)
    parser.add_argument('--learning_rate', '-r', help='Specify learning rate', type=float, default=0.001)
    parser.add_argument('--lr_scheduler', '-lrs', help='Specify schedule', type=int, nargs='*', action='store', default=[100, 200])
    parser.add_argument('--momentum', '-m', help='Momentum', type=float, default=0.5)
    parser.add_argument('--optimizer', '-o', help='Specify optimizer: SGD, Adam, RMSprop', type=str, default='SGD')
    parser.add_argument('--loss', '-l', help='Specify loss function: CE, FL', type=str, default='CE')
    parser.add_argument('--num_workers', '-w', help='Number of workers', type=int, default=8 )
    parser.add_argument('--device', '-d', help='Device', type=str, default='cuda:0')
    parser.add_argument('--use_tensorboard', '-tb', help='Use tensorboard', type=bool, default=True)
    args = parser.parse_args()

    main(**args.__dict__)