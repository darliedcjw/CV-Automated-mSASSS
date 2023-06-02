import torch
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
from tqdm import tqdm
from datetime import datetime

from loss import focal_loss
from model import ResNet152
from utils import save_checkpoint, compute_accuracy, compute_precision, compute_recall, compute_f1_score

torch.manual_seed(0)

class Train():
    
    def __init__(self,
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
        ):

        self.ds_train = ds_train
        self.ds_val = ds_val
        self.log_path = log_path
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.momentum = momentum
        self.num_workers = num_workers
        self.use_tensorboard = use_tensorboard

        # Device
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        print(self.device)

        
        # SummaryWriter
        self.log_path = os.path.join(self.log_path, datetime.now().strftime('%d%m%y_%H%M%S'))
        os.makedirs(self.log_path, exist_ok=False)
        if self.use_tensorboard:
            self.summary_writer = SummaryWriter(self.log_path)

        
        # Write Parameters
        self.parameters = [x + ':' + str(y) + '\n' for x, y in locals().items()]
        with open(os.path.join(self.log_path, 'parameters.txt'), 'w') as fd:
            fd.writelines(self.parameters)
        
        if self.use_tensorboard:
            self.summary_writer.add_text('parameters', '\n'.join(self.parameters))


        # Model
        self.model = ResNet152(in_channels=3, num_classes=num_classes).to(device)

        # Optimizer
        if optimizer == 'SGD':
            self.optimizer = SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif optimizer == 'Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer == 'RMSprop':
            self.optimizer = RMSprop(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        else:
            raise NotImplementedError('Please specify the correct optimizer!')        

        if self.lr_scheduler != None:
            self.scheduler = MultiStepLR(self.optimizer, milestones=lr_scheduler, gamma=0.1)

        # Loss
        if loss == 'CE':
            self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        elif loss == 'FL':
            self.loss_fn = focal_loss(wf=0.15, fp=2).to(self.device)


        # DataLoader
        self.dl_train = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=False)
        self.len_dl_train = len(self.dl_train)
        print('Training Data Loaded: {}'.format(self.len_dl_train))

        self.dl_val = DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=False)
        self.len_dl_val = len(self.dl_val)
        print('Validation Data Loaded {}'.format(self.len_dl_val))
    
    def _train(self):

        self.model.train()

        total_loss_train = 0
        total_acc_train = 0
        
        total_precision_train = 0
        total_precisioncount_train = 0

        total_recall_train = 0
        total_recallcount_train = 0

        for step, (image, label) in enumerate(tqdm(self.dl_train, desc='Training')):

            loss = 0

            image = image.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(image)

            loss = 0.5 * self.loss_fn(output, label)

            loss.backward()

            self.optimizer.step()

            acc = compute_accuracy(output, label, self.batch_size)

            total_loss_train += loss.item()
            total_acc_train += acc.item()
            
            precision, precision_count = compute_precision(output, label, self.num_classes)
            recall, recall_count = compute_recall(output, label, self.num_classes)
            
            total_precision_train += precision
            total_precisioncount_train += precision_count

            total_recall_train += recall
            total_recallcount_train += recall_count
            
            if self.use_tensorboard: # End of a batch
                self.summary_writer.add_scalar('train_loss', loss.item(), global_step=step + self.epoch * self.len_dl_train) 
                self.summary_writer.add_scalar('train_acc', acc.item(), global_step=step + self.epoch * self.len_dl_train)


        self.mean_loss_train = total_loss_train / self.len_dl_train
        self.mean_acc_train = total_acc_train / self.len_dl_train
        self.mean_precision_train = total_precision_train / total_precisioncount_train
        self.mean_recall_train = total_recall_train / total_recallcount_train
        self.mean_f1_train = compute_f1_score(self.mean_precision_train, self.mean_recall_train)

        print('Train: \nLoss: {0}\n Accuracy: {1}\n Precision: {2}\n Recall: {3}\n F1 Score: {4}\n'
              .format(self.mean_loss_train,
                      self.mean_acc_train,
                      self.mean_precision_train,
                      self.mean_recall_train,
                      self.mean_f1_train
                    )
            )


    def _val(self):

        self.model.eval()

        total_loss_val = 0
        total_acc_val = 0
        
        total_precision_val = 0
        total_precisioncount_val = 0

        total_recall_val = 0
        total_recallcount_val = 0

        with torch.no_grad():
            for step, (image, label) in enumerate(tqdm(self.dl_val, desc='Validating')):
                
                loss = 0

                image = image.to(self.device)
                label = label.to(self.device)

                output = self.model(image)

                loss =  0.5 * self.loss_fn(output, label)

                acc = compute_accuracy(output, label, self.batch_size)

                total_loss_val += loss.item()
                total_acc_val += acc.item()

                precision, precision_count = compute_precision(output, label, self.num_classes)
                recall, recall_count = compute_recall(output, label, self.num_classes)      
                
                total_precision_val += precision
                total_precisioncount_val += precision_count
                total_recall_val += recall
                total_recallcount_val += recall_count

                if self.use_tensorboard: # End of a batch
                    self.summary_writer.add_scalar('val_loss', loss.item(), global_step=step + self.epoch * self.len_dl_val) 
                    self.summary_writer.add_scalar('val_acc', acc.item(), global_step=step + self.epoch * self.len_dl_val)
            
            self.mean_loss_val = total_loss_val / self.len_dl_val
            self.mean_acc_val = total_acc_val / self.len_dl_val
            self.mean_precision_val = total_precision_val / total_precisioncount_val
            self.mean_recall_val = total_recall_val / total_recallcount_val
            self.mean_f1_val = compute_f1_score(self.mean_precision_val, self.mean_recall_val)

            
            print('Validation: \nLoss: {0}\n Accuracy: {1}\n Precision: {2}\n Recall: {3}\n F1 Score: {4}\n'
                .format(self.mean_loss_val,
                        self.mean_acc_val,
                        self.mean_precision_val,
                        self.mean_recall_val,
                        self.mean_f1_val
                        )
                )

    def _checkpoint(self):

        save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_last.pth'), epoch=self.epoch + 1, 
                            model=self.model, optimizer=self.optimizer, params=self.parameters)
        
        if self.best_loss is None or self.best_acc <= self.mean_acc_val:
            self.best_loss = self.mean_loss_val
            self.best_acc = self.mean_acc_val
            print('best metrics: loss - {0:.4f}, acc - {1:.4f} at epoch {2}'.format(self.best_loss, self.best_acc, self.epoch + 1))
            
            save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_best_{0:.4f}_{1:.4f}.pth'.format(self.best_loss, self.best_acc)), epoch=self.epoch + 1,
                            model=self.model, optimizer=self.optimizer, params=self.parameters)
            
            with open(os.path.join(self.log_path, 'metrics'), 'a+') as f:
                    f.seek(0)
                    data = f.read(100)
                    if len(data) > 0:
                        f.write('\n')
                    f.write(
                        'Epoch: {0}, Loss: {1:.4f}, Accuracy: {2:.4f}, Precision: {3}, Recall: {4}, F1 Score: {5}'
                        .format(self.epoch + 1,
                                self.best_loss,
                                self.best_acc,
                                self.mean_precision_val,
                                self.mean_recall_val,
                                self.mean_f1_val))
                    
            
    def run(self):

        print('\nTraining started at {}'.format(datetime.now().strftime('%T-%m-%d %H:%M:%S')))

        self.best_loss = None
        self.best_acc = None

        for self.epoch in range(self.epochs):
            
            self.mean_loss_train = 0
            self.mean_loss_val = 0
            
            self.mean_acc_train = 0
            self.mean_acc_val = 0

            self.mean_precision_train = 0
            self.mean_precision_val = 0

            self.mean_recall_train = 0
            self.mean_recall_val = 0
            
            self.mean_f1_train = 0
            self.mean_f1_val = 0

            print('\nEpoch: {}'.format(self.epoch + 1))

            self._train()

            self._val()

            self._checkpoint()

            if self.lr_scheduler != None:
                self.scheduler.step()

        if self.use_tensorboard:
            self.summary_writer.close()
        
        print('\nTraining ended at {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))