import ast
from datetime import datetime
import argparse

from datasets import COCODataset
from train import Train


import torch


def main(exp_name,
         epochs=210,
         batch_size=4,
         num_workers=2,
         disable_dist = False,
         lr=0.001,
         disable_lr_decay=False,
         lr_decay_steps='(170, 200)',
         lr_decay_gamma=0.1,
         optimizer='Adam',
         weight_decay=0.0,
         momentum=0.9,
         nesterov=False,
         pretrained_weight_path=None,
         log_path='./logs',
         disable_tensorboard_log=False,
         model_c=32,
         model_key=12,
         model_bn_momentum=0.1,
         image_resolution='(256, 192)',
         coco_root_path="./datasets/COCO",
         device=None,
         disable_seed=False):

    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    print(device)

    print("\nStarting experiment `%s` @ %s\n" % (exp_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    use_seed = not disable_seed
    use_dist = not disable_dist
    lr_decay = not disable_lr_decay
    use_tensorboard = not disable_tensorboard_log
    image_resolution = ast.literal_eval(image_resolution)
    lr_decay_steps = ast.literal_eval(lr_decay_steps)

    if use_seed:
        torch.manual_seed(0)

    print("\nLoading train and validation datasets...")

    # load train and val datasets
    ds_train = COCODataset(
        root_path=coco_root_path, data_version="default", keys=model_key, use_dist=use_dist, is_train=True, is_rotate=False, is_flip=False, is_scale=False, image_width=image_resolution[1], image_height=image_resolution[0], color_rgb=True)
    print('Training Size: {}'.format(ds_train.__len__()))

    ds_val = COCODataset(
        root_path=coco_root_path, data_version="default_val", keys=model_key, use_dist=use_dist, is_train=False, image_width=image_resolution[1], image_height=image_resolution[0], color_rgb=True)
    print('Validation Size: {}'.format(ds_val.__len__()))

    train = Train(
        exp_name=exp_name,
        ds_train=ds_train,
        ds_val=ds_val,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        loss='JointsMSELoss',
        use_dist=use_dist,
        lr=lr,
        lr_decay=lr_decay,
        lr_decay_steps=lr_decay_steps,
        lr_decay_gamma=lr_decay_gamma,
        optimizer=optimizer,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        pretrained_weight_path=pretrained_weight_path,
        log_path=log_path,
        use_tensorboard=use_tensorboard,
        model_c=model_c,
        model_key=model_key,
        model_bn_momentum=model_bn_momentum,
        device=device
    )

    train.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n",
                        help="experiment name. A folder with this name will be created in the log_path.",
                        type=str, default=str(datetime.now().strftime("%Y%m%d_%H%M")))
    parser.add_argument("--epochs", "-e", help="number of epochs", type=int, default=210)
    parser.add_argument("--batch_size", "-b", help="batch size", type=int, default=4)
    parser.add_argument("--num_workers", "-w", help="number of DataLoader workers", type=int, default=2)
    parser.add_argument("--disable_dist", "-dd", help="disable training distance loss", action="store_true")
    parser.add_argument("--lr", "-l", help="initial learning rate", type=float, default=0.001)
    parser.add_argument("--disable_lr_decay", help="disable learning rate decay", action="store_true")
    parser.add_argument("--lr_decay_steps", help="learning rate decay steps", type=str, default="(170, 200)")
    parser.add_argument("--lr_decay_gamma", help="learning rate decay gamma", type=float, default=0.1)
    parser.add_argument("--optimizer", "-o", help="optimizer name. Currently, only `SGD` and `Adam` are supported.",
                        type=str, default='Adam')
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=0.0)
    parser.add_argument("--momentum", "-m", help="momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", help="enable nesterov", action="store_true")
    parser.add_argument("--pretrained_weight_path", "-p",
                        help="pre-trained weight path. Weights will be loaded before training starts.",
                        type=str, default=None)
    parser.add_argument("--log_path", help="log path. tensorboard logs and checkpoints will be saved here.",
                        type=str, default='./logs')
    parser.add_argument("--disable_tensorboard_log", "-u", help="disable tensorboard logging", action="store_true")
    parser.add_argument("--model_c", help="HRNet c parameter", type=int, default=32)
    parser.add_argument("--model_key", help="HRNet key parameter", type=int, default=12)
    parser.add_argument("--model_bn_momentum", help="HRNet bn_momentum parameter", type=float, default=0.1)
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(256, 192)')
    parser.add_argument("--coco_root_path", help="COCO dataset root path", type=str, default="./datasets/COCO")
    parser.add_argument("--device", "-d", help="device", type=str, default=None)
    parser.add_argument("--disable_seed", "-ds", help="disable seed", action="store_true")
    args = parser.parse_args()

    main(**args.__dict__)       