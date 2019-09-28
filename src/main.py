import time
kernel_start_time = time.perf_counter()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import SVG, display

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os, time, glob, shutil
starttime = time.time()

import warnings
warnings.simplefilter("ignore")

from pathlib import Path
import xml.etree.ElementTree as ET # for parsing XML
from PIL import Image # to read images
import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from argparse import Namespace
import numpy as np
import random
import torch
from torch import nn, optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.utils as vutils

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

import data
import model
from utils import *
from args import args


seed_everything(seed=args.seed)
transform1, transform2 = data.get_transforms(args.image_size)
train_data = data.DogDataset(img_dir=args.root_images,
							 args=args,
		                     transform1=transform1,
	    	                 transform2=transform2)
decoded_dog_labels = {i:breed for i, breed in enumerate(sorted(set(train_data.labels)))}
encoded_dog_labels = {breed:i for i, breed in enumerate(sorted(set(train_data.labels)))}
train_data.labels = [encoded_dog_labels[l] for l in train_data.labels] # encode dog labels in the data generator
dataloader = torch.utils.data.DataLoader(train_data, 
                                         shuffle=True,
                                         batch_size=args.batch_size, 
                                         num_workers=args.num_workers)
netG = model.Generator(args).to(device)
netD = model.Discriminator(args).to(device)
model.weights_init(netG)
model.weights_init(netD)
print("Generator parameters:    ", sum(p.numel() for p in netG.parameters() if p.requires_grad))
print("Discriminator parameters:", sum(p.numel() for p in netD.parameters() if p.requires_grad))

fixed_noise = torch.randn(64, args.latent_dim, device=device)
dog_labels = torch.randint(0, len(encoded_dog_labels), (64, ), device=device)

# Establish convention for real and fake labels during training
real_label = 0.9
fake_label = 0.0

BCE_stable = nn.BCEWithLogitsLoss()
optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))


def step(engine, batch):
    def train_D(images, labels):
        """
        Run 1 step of training for discriminator
        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: non-saturing loss for discriminator,
            -E[log( sigmoid(D(x) - E[D(G(z))]) )]
              - E[log(1 - sigmoid(D(G(z)) - E[D(x)]))]
        """
        noise = torch.randn(args.batch_size, args.latent_dim, device=device)
        G_output = netG(noise, labels)
        
        # classify the generated and real batch images
        Dx_score = netD(images, labels) # D(x)
        DG_score = netD(G_output, labels) # D(G(z))
        
        # Compute RA Hinge loss
        D_loss = (torch.mean(torch.nn.ReLU()(1.0 - (Dx_score - torch.mean(DG_score)))) + 
                  torch.mean(torch.nn.ReLU()(1.0 + (DG_score - torch.mean(Dx_score)))))/2
        
        return D_loss, F.sigmoid(Dx_score).mean().item(), F.sigmoid(DG_score).mean().item()
    
    def train_G(images, labels):
        """ Run 1 step of training for generator
            Input:
                images: batch of images reshaped to [batch_size, -1]
            Output:
                G_loss: non-saturating loss for how well G(z) fools D,
                -E[log(sigmoid(D(G(z))-E[D(x)]))]
                    -E[log(1-sigmoid(D(x)-E[D(G(z))]))]
        """
        noise = torch.randn(args.batch_size, args.latent_dim, device=device)
        G_output = netG(noise, labels) # G(z)
                  

        Dx_score = netD(images, labels) # D(x)
        DG_score = netD(G_output, labels) # D(G(z))
        
        # Compute RA NS loss for G                
        G_loss = (torch.mean(torch.nn.ReLU()(1.0 + (Dx_score - torch.mean(DG_score)))) + 
                  torch.mean(torch.nn.ReLU()(1.0 - (DG_score - torch.mean(Dx_score)))))/2
        
        return G_loss, F.sigmoid(DG_score).mean().item()
    
    images, labels = batch[0].to(device), batch[1].to(device)
    args.batch_size = images.size(0)
    D_step_loss = []
    for _ in range(args.num_disc_update):
        netD.zero_grad()
        
        D_loss, Dx_score, DG_score1 = train_D(images, labels)
        D_loss.backward()
        optimizerD.step()
        D_step_loss.append(D_loss.item())
                    
    # update G
    netG.zero_grad()

    G_loss, DG_score2 = train_G(images, labels)
    G_loss.backward()
    optimizerG.step()
        
    return {
            'D_loss': np.mean(D_step_loss),
            'G_loss': G_loss.item(),
            'Dx_score': Dx_score,
            'DG_score1': DG_score1,
            'DG_score2': DG_score2
    }

clear_output_dir(args)

# ignite objects
trainer = Engine(step)
checkpoint_handler = ModelCheckpoint(args.output_dir, args.CKPT_PREFIX, save_interval=1, n_saved=10, require_empty=False)
timer = Timer(average=True)

# attach running average metrics
monitoring_metrics = ['D_loss', 'G_loss', 'Dx_score', 'DG_score1', 'DG_score2']
RunningAverage(alpha=args.alpha, output_transform=lambda x: x['D_loss']).attach(trainer, 'D_loss')
RunningAverage(alpha=args.alpha, output_transform=lambda x: x['G_loss']).attach(trainer, 'G_loss')
RunningAverage(alpha=args.alpha, output_transform=lambda x: x['Dx_score']).attach(trainer, 'Dx_score')
RunningAverage(alpha=args.alpha, output_transform=lambda x: x['DG_score1']).attach(trainer, 'DG_score1')
RunningAverage(alpha=args.alpha, output_transform=lambda x: x['DG_score2']).attach(trainer, 'DG_score2')

# attach progress bar
pbar = ProgressBar()
pbar.attach(trainer, metric_names=monitoring_metrics)

# adding handlers using `trainer.on` decorator API
@trainer.on(Events.ITERATION_COMPLETED)
def print_logs(engine):
    if (engine.state.iteration - 1) % args.PRINT_FREQ == 0:
        fname = os.path.join(args.output_dir, args.LOGS_FNAME)
        columns = ["iteration", ] + list(engine.state.metrics.keys())
        values = [str(engine.state.iteration), ] + \
                 [str(round(value, 5)) for value in engine.state.metrics.values()]
        with open(fname, 'a') as f:
            if f.tell() == 0:
                print('\t'.join(columns), file=f)
            print('\t'.join(values), file=f)

        message = '[{epoch}/{max_epoch}][{i}/{max_i}]'.format(epoch=engine.state.epoch,
                                                              max_epoch=args.num_epochs,
                                                              i=(engine.state.iteration % len(dataloader)),
                                                              max_i=len(dataloader))
        for name, value in zip(columns, values):
            message += ' | {name}: {value}'.format(name=name, value=value)
        pbar.log_message(message)

@trainer.on(Events.EPOCH_COMPLETED)
def save_fake_example(engine):
    fake = netG(fixed_noise, dog_labels).detach().cpu()
    path = os.path.join(args.output_dir, args.FAKE_IMG_FNAME.format(engine.state.epoch))
    vutils.save_image((fake+1.)/2., path, normalize=True)
    
@trainer.on(Events.EPOCH_COMPLETED)
def save_real_example(engine):
    img, _ = engine.state.batch
    path = os.path.join(args.output_dir, args.REAL_IMG_FNAME.format(engine.state.epoch))
    vutils.save_image(img, path, normalize=True)

@trainer.on(Events.EPOCH_COMPLETED)
def display_images(engine):
    if engine.state.epoch % 10 == 0:
        display(Image.open(os.path.join(args.output_dir, args.REAL_IMG_FNAME.format(engine.state.epoch))))
        display(Image.open(os.path.join(args.output_dir, args.FAKE_IMG_FNAME.format(engine.state.epoch))))    
    
# adding handlers using `trainer.add_event_handler` method API
trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, 
                          handler=checkpoint_handler,
                          to_save={
                              'netG': netG,
                              'netD': netD
                          })

# automatically adding handlers via a special `attach` method of `Timer` handler
timer.attach(trainer, 
             start=Events.EPOCH_STARTED, 
             resume=Events.EPOCH_STARTED,
             pause=Events.EPOCH_COMPLETED, 
             step=Events.EPOCH_COMPLETED)

@trainer.on(Events.EPOCH_COMPLETED)
def print_times(engine):
    pbar.log_message(f'Epoch {engine.state.epoch} done. Time per epoch: {timer.value()/60:.3f}[min]')
    timer.reset()
    
@trainer.on(Events.EPOCH_COMPLETED)
def create_plots(engine):
    try:
        import matplotlib as mpl
        mpl.use('agg')

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

    except ImportError:
        warnings.warn('Loss plots will not be generated -- pandas or matplotlib not found')

    else:
        df = pd.read_csv(os.path.join(args.output_dir, args.LOGS_FNAME), delimiter='\t', index_col='iteration')
        _ = df.loc[:, list(engine.state.metrics.keys())].plot(subplots=True, figsize=(10, 10))
        _ = plt.xlabel('Iteration number')
        fig = plt.gcf()
        path = os.path.join(args.output_dir, args.PLOT_FNAME)
        fig.savefig(path)
        
@trainer.on(Events.EPOCH_STARTED)
def handle_timeout(engine):
    if time.perf_counter() - kernel_start_time > 45000:
        print("Time limit reached! Stopping kernel!")
        engine.terminate()

        create_plots(engine)
        checkpoint_handler(engine, {
            'netG_exception': netG,
            'netD_exception': netD
        })
        
@trainer.on(Events.EXCEPTION_RAISED)
def handle_exception(engine, e):
    if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
        engine.terminate()
        warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')

        create_plots(engine)
        checkpoint_handler(engine, {
            'netG_exception': netG,
            'netD_exception': netD
        })

    else:
        raise e
        
trainer.run(dataloader, args.num_epochs)


good_breeds = analyse_generated_by_class(6, decoded_dog_labels)
create_submit(good_breeds)
