import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from forward.simple_model import SimpleLayerModel, SimpleLayerDataset

from utils.data_vis import plot_speeds, plot_amplitudes


class Callback():
    def __init__(self): pass
    def on_train_begin(self,s): pass
    def on_train_end(self,s): pass
    def on_epoch_begin(self,s): pass
    def on_epoch_end(self,s): pass
    def on_batch_begin(self,s): pass
    def on_batch_end(self,s): pass
    def on_val_begin(self,s): pass
    def on_val_end(self,s): pass


class TBWriter(Callback):
    
    def on_train_begin(self,s):
        
        self.writer = SummaryWriter(comment=f'LR_{s.lr}_BS_{s.batch_size}')
        
    def on_train_end(self,s): 
        self.writer.close()

    def on_batch_end(self,s):
        self.writer.add_scalar('Loss/train', s.batch_loss, s.global_step)
        
        
        lr = s.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('LR',lr,s.global_step)
        
        
    def on_val_end(self,s):
                
        amps_plot = plot_amplitudes(s.current_X)
        
        self.writer.add_images('images', amps_plot, s.global_step)
        
        speeds_plot = plot_speeds(s.current_Y.detach().cpu().numpy()[0],
                                  s.current_Y_pred.detach().cpu().numpy()[0])

        self.writer.add_images('speeds', speeds_plot, s.global_step)
        

class TrainState():
    
    def __init__(self,
                 model,
                 train_dataset,
                 val_dataset,                 
                 n_epochs = 5,
                 lr=0.1,
                 batch_size = 1,
                 n_subbatches = 1,
                 val_interval = 500,
                 save_dir = None,
                 device = None,
                 optimizer = None,
                 criterion = None,
                 ):
           

        if not device:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        if not optimizer:
            optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
        
        if not criterion:
            criterion = nn.MSELoss() 
            
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5, pin_memory=True, worker_init_fn=lambda x: np.random.seed())
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=5, pin_memory=True)

        self.n_train = len(train_dataset)
        self.n_val = len(val_dataset)

        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_subbatches = n_subbatches
        self.val_interval = val_interval
        self.save_dir = None

        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        # State
        self.batch_num = 0
        self.global_step = 0
        self.epoch_loss = 0
        self.val_loss = 0
        self.batch_loss = 0
        self.current_X = None
        self.current_Y = None
        self.current_Y_pred = None

def train_net(
        state,
        callbacks
              ):
    
    s = state
    model = s.model
    criterion = s.criterion
    optimizer = s.optimizer
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logging.info(f'Network:\n'
             f'\t{model.n_channels} input channels\n')
    
    logging.info(f'''Starting training:
        Epochs:          {s.n_epochs}
        Batch size:      {s.batch_size}
        Learning rate:   {s.lr}
        Training size:   {s.n_train}
        Validation size: {s.n_val}
        Device:          {s.device.type}
    ''')

    model.to(device=s.device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    model.train()

    # Train Start Callback
    for c in callbacks: c.on_train_begin(s)
    
    for epoch in range(s.n_epochs):
        
        # Epoch Begin Callback
        for c in callbacks: c.on_epoch_begin(s)
        
        loss = 0
        
        with tqdm(total=np.ceil(s.n_train/s.batch_size), desc=f'Epoch {epoch + 1}/{s.n_epochs}', unit='img') as pbar:
            
            for (k,batch) in enumerate(s.train_loader):
                
                # Batch Begin Callback
                for c in callbacks: c.on_batch_begin(s)

                assert batch['X'].shape[1] == model.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded amplitudes have {imgs.shape[1]} channels. Please check that ' \
                    'the net is configured correctly.'
        
                X = batch['X'].to(device=s.device, dtype=torch.float32)
                Y = batch['Y'].to(device=s.device, dtype=torch.float32)
                Y_pred = model(X)
                
                s.current_X = X
                s.current_Y = Y
                s.current_Y_pred = Y_pred
                
                loss += criterion(Y_pred, Y)


                if k % s.n_subbatches == 0:

                    s.epoch_loss += loss.item()
                    s.batch_loss = loss.item()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss = 0

                    pbar.set_postfix(**{'loss (batch)': s.batch_loss})
                    
                    # Batch End Callbacks
                    for c in callbacks: c.on_batch_end(s)

                
                pbar.update()
                s.global_step += 1
                
                if s.global_step % s.val_interval == 1:
                    
                    #val_score = eval_net(net, val_loader, device, n_val)
                    #net.train()
                    
                    #logging.info('Validation cross entropy: {}'.format(val_score))
                    #writer.add_scalar('Loss/test', val_score, global_step)

                    for c in callbacks: c.on_val_end(s)

        # Epoch End Callback
        for c in callbacks: c.on_epoch_end(s)
        
        if s.save_dir:
            try:
                os.mkdir(s.save_dir)
            except OSError:
                pass
            torch.save(net.state_dict(),
                       s.save_dir + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
            

    # Train End Callback
    for c in callbacks: c.on_train_end(s)  
        
