import os
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from HRnet import HRNet
from loss import JointsMSELoss
from misc.visualization import save_images
from misc.checkpoint import save_checkpoint

class Train():

    def __init__(self,
        exp_name,
        ds_train,
        ds_val,
        epochs=210,
        batch_size=4,
        num_workers=2,
        loss='JointsMSELoss',
        use_dist=True,
        lr=0.001,
        lr_decay=True,
        lr_decay_steps=[170, 200],
        lr_decay_gamma=0.1,
        optimizer='Adam',
        weight_decay=0.,
        momentum=0.9,
        nesterov=False,
        pretrained_weight_path=None,
        checkpoint_path=None,
        log_path='./logs',
        use_tensorboard=True,
        model_c=32,
        model_key=12,
        model_bn_momentum=0.1,
        device=None
        ):

        # torch.manual_seed(0)
        self.exp_name = exp_name
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loss = loss
        self.use_dist = use_dist
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_gamma = lr_decay_gamma
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.pretrained_weight_path = pretrained_weight_path
        self.checkpoint_path = checkpoint_path
        self.log_path = os.path.join(log_path, self.exp_name)
        self.use_tensorboard = use_tensorboard
        self.model_c = model_c
        self.model_key = model_key
        self.model_bn_momentum = model_bn_momentum
        self.epoch = 0

        # torch device
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        print(self.device)

        os.makedirs(self.log_path, 0o755, exist_ok=False)  # exist_ok=False to avoid overwriting
        if self.use_tensorboard:
            self.summary_writer = SummaryWriter(self.log_path)

        # write all experiment parameters in parameters.txt and in tensorboard text field
        self.parameters = [x + ': ' + str(y) + '\n' for x, y in locals().items()]
        with open(os.path.join(self.log_path, 'parameters.txt'), 'w') as fd:
            fd.writelines(self.parameters)
        if self.use_tensorboard:
            self.summary_writer.add_text('parameters', '\n'.join(self.parameters))

        # Load Pre-Train Model
        self.model = HRNet(c=self.model_c, key=17).to(self.device)

        # Loss
        if self.loss == 'JointsMSELoss':
            self.loss_fn = JointsMSELoss(use_point_dist=self.use_dist).to(self.device) # Include Non-visible Points
        else:
            raise NotImplementedError

        # Optimizer
        if optimizer == 'SGD':
            self.optim = SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                momentum=self.momentum, nesterov=self.nesterov)
        elif optimizer == 'Adam':
            self.optim = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

        # LR Decay
        if lr_decay:
            self.lr_scheduler = MultiStepLR(self.optim, list(self.lr_decay_steps), gamma=self.lr_decay_gamma)

        # load pre-trained weights (such as those pre-trained on imagenet)
        if self.pretrained_weight_path is not None:
            missing_keys, unexpected_keys = self.model.load_state_dict(
                torch.load(self.pretrained_weight_path, map_location=self.device),
                strict=False  # strict=False is required to load models pre-trained on imagenet
            )
            print('Pre-trained weights loaded.')
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                print('Pre-trained weights missing keys:', missing_keys)
                print('Pre-trained weights unexpected keys:', unexpected_keys)

        # Adapting to 12 keys instead of 17 keys
        self.model.final_layer = nn.Conv2d(self.model_c, self.model_key, kernel_size=1, stride=1).to(self.device)
        self.model.final_layer.weight.data = torch.normal(mean=0, std=1, size=(self.model_key, self.model_c, 1, 1), requires_grad=True).to(self.device)
        print('Transformation done for {} keypoints'.format(self.model_key))

        # Train Loader
        self.dl_train = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        self.len_dl_train = len(self.dl_train)

        # Val Loader
        self.dl_val = DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        self.len_dl_val = len(self.dl_val)

        # initialize variables
        self.mean_loss_train = 0.
        self.mean_acc_train = 0.
        self.mean_loss_val = 0.
        self.mean_acc_val = 0.
        self.mean_mAP_val = 0.

        self.best_loss = None
        self.best_acc = None
        self.best_mAP = None


    def _train(self):

        # Switch model to training mode
        self.model.train()


        for step, (image, target, target_weight, points_data) in enumerate(tqdm(self.dl_train, desc='Training')):
            # For each batch
            image = image.to(self.device)
            target = target.to(self.device)
            target_weight = target_weight.to(self.device)
            points_distance = points_data['points_distance'].to(self.device)
            points_vis = points_data['points_visibility'].to(self.device)
            points_scale = points_data['scale'].to(self.device)
            
             ### Define everything here first

            self.optim.zero_grad()

            output = self.model(image) # Tensors

            # Loss for each batch
            loss = self.loss_fn(output, target, points_distance, points_vis, points_scale, self.device, target_weight)

            loss.backward()

            self.optim.step()

            # Evaluate accuracy
            # Get predictions on the input
            accs, avg_acc, cnt, points_preds, points_target = self.ds_train.evaluate_accuracy(output, target)

            self.mean_loss_train += loss.item()
            self.mean_acc_train += avg_acc.item()

            if self.use_tensorboard:
                self.summary_writer.add_scalar('train_loss', loss.item(),
                                               global_step=step + self.epoch * self.len_dl_train)
                self.summary_writer.add_scalar('train_acc', avg_acc.item(),
                                               global_step=step + self.epoch * self.len_dl_train)
                if step == 0:
                    save_images(image, target, points_target, output, points_preds, points_data['points_visibility'],
                                self.summary_writer, step=step + self.epoch * self.len_dl_train, prefix='train_')

                self.summary_writer.flush()

        self.mean_loss_train /= len(self.dl_train)
        self.mean_acc_train /= len(self.dl_train)

        print('\nTrain: Loss %f - Accuracy %f' % (self.mean_loss_train, self.mean_acc_train))


    def _val(self):
        self.model.eval()

        with torch.no_grad():
            for step, (image, target, target_weight, points_data) in enumerate(tqdm(self.dl_val, desc='Validating')):
                image = image.to(self.device)
                target = target.to(self.device)
                target_weight = target_weight.to(self.device)
                points_distance = points_data['points_distance'].to(self.device)
                points_vis = points_data['points_visibility'].to(self.device)
                points_scale = points_data['scale'].to(self.device)

                output = self.model(image)

                loss = self.loss_fn(output, target, points_distance, points_vis, points_scale, self.device, target_weight)

                _, avg_acc, _, points_preds, points_target = \
                    self.ds_val.evaluate_accuracy(output, target)

                self.mean_loss_val += loss.item()
                self.mean_acc_val += avg_acc.item()
                if self.use_tensorboard:
                    self.summary_writer.add_scalar('val_loss', loss.item(),
                                                   global_step=step + self.epoch * self.len_dl_val)
                    self.summary_writer.add_scalar('val_acc', avg_acc.item(),
                                                   global_step=step + self.epoch * self.len_dl_val)
                    if step == 0:
                        save_images(image, target, points_target, output, points_preds,
                                    points_data['points_visibility'], self.summary_writer,
                                    step=step + self.epoch * self.len_dl_val, prefix='val_')
                    
                    self.summary_writer.flush()

        self.mean_loss_val /= len(self.dl_val)
        self.mean_acc_val /= len(self.dl_val)

        print('\nValidation: Loss %f - Accuracy %f' % (self.mean_loss_val, self.mean_acc_val))

    
    def _checkpoint(self):

        save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_last.pth'), epoch=self.epoch + 1, 
                            model=self.model, optimizer=self.optim, params=self.parameters)
        
        if self.best_loss is None or self.best_acc <= self.mean_acc_val:
            self.best_loss = self.mean_loss_val
            self.best_acc = self.mean_acc_val
            print('best metrics: loss - {0:.4f}, acc - {1:.4f} at epoch {2}'.format(self.best_loss, self.best_acc, self.epoch + 1))
            
            save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_best_{0:.4f}_{1:.4f}.pth'.format(self.best_loss, self.best_acc)), epoch=self.epoch + 1,
                            model=self.model, optimizer=self.optim, params=self.parameters)
            
            with open(os.path.join(self.log_path, 'metrics'), 'a+') as f:
                    f.seek(0)
                    data = f.read(100)
                    if len(data) > 0:
                        f.write('\n')
                    f.write(
                        'Epoch: {0}, Validation Loss: {1:.4f}, Valdation Accuracy: {2:.4f}, Training Loss: {3:.4f}, Training Accuracy: {4:.4f}'
                        .format(self.epoch + 1,
                                self.best_loss,
                                self.best_acc,
                                self.mean_loss_train,
                                self.mean_acc_train))
    def run(self):
        """
        Runs the training.
        """

        print('\nTraining started @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # start training
        for self.epoch in range(self.epochs):
            print('\nEpoch %d of %d @ %s' % (self.epoch + 1, self.epochs, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            self.mean_loss_train = 0.
            self.mean_loss_val = 0.
            self.mean_acc_train = 0.
            self.mean_acc_val = 0.
            self.mean_mAP_val = 0.

            self._train()

            self._val()

            if self.lr_decay:
                self.lr_scheduler.step()

            self._checkpoint()

        if self.use_tensorboard:
            self.summary_writer.close()
            
        print('\nTraining ended @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))