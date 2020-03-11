import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm_notebook as tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from forward.simple_model import SimpleLayerModel, SimpleLayerDataset

from utils.data_vis import plot_speeds, plot_amplitudes


def train_net(net,
              train_dataset,
              val_dataset,
              device = None,
              epochs=5,
              batch_size=1,
              n_subbatches=1,
              lr=0.1,
              optimizer = None,
              scheduler = None,
              val_interval = 500,
              save_dir=None,
              callbacks
              ):
    
    
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5, pin_memory=True, worker_init_fn=lambda x: np.random.seed())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=5, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0
    
    if device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    if optimizer is None:
        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
            
    criterion = nn.MSELoss()
    
    net.train()
    
    for epoch in range(epochs):
        
        epoch_loss = 0
        loss = 0
        
        with tqdm(total=np.ceil(n_train/batch_size), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            
            for (k,batch) in enumerate(train_loader):
        
                imgs = batch['amplitudes']
                speeds = batch['speeds']

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded amplitudes have {imgs.shape[1]} channels. Please check that ' \
                    'the net is configured correctly.'
        
                imgs = imgs.to(device=device, dtype=torch.float32)
                speeds = speeds.to(device=device, dtype=torch.float32).squeeze()
        
                speeds_pred = net(imgs)
                loss += criterion(speeds_pred.squeeze(), speeds.squeeze())
                
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                if k % n_subbatches == 0:
    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                pbar.update()
                global_step += 1
                
                if global_step % val_interval == 1:
                    
                    #val_score = eval_net(net, val_loader, device, n_val)
                    #net.train()
                    
                    #logging.info('Validation cross entropy: {}'.format(val_score))
                    #writer.add_scalar('Loss/test', val_score, global_step)

                    amps_plot = plot_amplitudes(imgs)
                    
                    writer.add_images('images', amps_plot, global_step)
                    
                    speeds_plot = plot_speeds(speeds.detach().cpu().numpy()[0],
                                              speeds_pred.detach().cpu().numpy()[0].squeeze())

                    writer.add_images('speeds', speeds_plot, global_step)
         
        if save_dir:
            try:
                os.mkdir(save_dir)
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
            
         # TODO: Add contingency for when validation loss is used in scheduler           
         if scheduler:
            scheduler.step()

    writer.close()

        
