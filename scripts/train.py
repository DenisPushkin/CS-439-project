import numpy
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import time
import copy
import json
import numpy as np
from plot import get_plot_func
from dataloader import load_mnist
from models import GeneratorCNN28, DiscriminatorCNN28
from ExtraAdam import ExtraAdam
from lookahead import Lookahead


def get_disciminator_loss(D, x_real, x_gen, lbl_real, lbl_fake):
  """"""
  D_x = D(x_real)
  D_G_z = D(x_gen)
  #print('Inside get_disciminator_loss')
  #print('D_x size =', D_x.size())
  #print('lbl_real size =', lbl_real.size())
  lossD_real = torch.binary_cross_entropy_with_logits(D_x, lbl_real).mean()
  lossD_fake = torch.binary_cross_entropy_with_logits(D_G_z, lbl_fake).mean()
  lossD = lossD_real + lossD_fake
  return lossD


def get_generator_loss(G, D, z, lbl_real):
  """"""
  D_G_z = D(G(z))
  lossG = torch.binary_cross_entropy_with_logits(D_G_z, lbl_real).mean()
  return lossG


def update_avg_gen(G, G_avg, n_gen_update):
    """ Updates the uniform average generator. """
    l_param = list(G.parameters())
    l_avg_param = list(G_avg.parameters())
    if len(l_param) != len(l_avg_param):
        raise ValueError("Got different lengths: {}, {}".format(len(l_param), len(l_avg_param)))

    for i in range(len(l_param)):
        with torch.no_grad():
            l_avg_param[i].data.copy_(l_avg_param[i].data.mul(n_gen_update).div(n_gen_update + 1.).add(
                                      l_param[i].data.div(n_gen_update + 1.)))

def update_ema_gen(G, G_ema, beta_ema=0.9999):
    """ Updates the exponential moving average generator. """
    l_param = list(G.parameters())
    l_ema_param = list(G_ema.parameters())
    if len(l_param) != len(l_ema_param):
        raise ValueError("Got different lengths: {}, {}".format(len(l_param), len(l_ema_param)))

    for i in range(len(l_param)):
        with torch.no_grad():
            l_ema_param[i].data.copy_(l_ema_param[i].data.mul(beta_ema).add(
                l_param[i].data.mul(1-beta_ema)))


def train(args, out_dir, pretrained_clf_path):

  epochs = args['epochs']
  batch_size = args['batch_size']
  eval_avg = args['eval_avg']
  n_workers = args['n_workers']
  device = args['device']
  grad_max_norm = args['grad_max_norm']
  seed = args['grad_max_norm']
  continue_training = args['continue_training']

  current_dir = get_dir(args, out_dir)
  if not os.path.isdir(current_dir):
    os.makedirs(current_dir, exist_ok=True)

  dataset = load_mnist(_data_root='datasets', binarized=False)
  dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True, drop_last=True)
  
  plot_func = get_plot_func(dataset=dataset,
                            pretrained_clf_path=pretrained_clf_path,
                            out_dir=current_dir, 
                            img_size=dataset[0][0].size(),
                            num_samples_eval=10000)
  
  G = GeneratorCNN28()
  D = DiscriminatorCNN28(img_size=28)
  G.to(device)
  D.to(device)

  G_avg, G_ema = None, None
  if eval_avg:
    G_avg = copy.deepcopy(G)
    G_ema = copy.deepcopy(G)

  optimizerG = get_optimizer(G, args)
  optimizerD = get_optimizer(D, args)

  curves = {
        'inception_means': [],
        'inception_stds': [],
        'inception_means_ema': [],
        'inception_means_avg': [],
        'fids_ema': [],
        'fids_avg': [],
        'fids': [],
        'epoch': 0,
        'batch': 0, #batches in total during all epochs; need to update G_avg
        'times': []
    }
  
  if continue_training:
    load_models(G, D, G_avg, G_ema, optimizerG, optimizerD, current_dir, suffix='last')
    curves = load_curves(current_dir)

  if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)

  # labels
  lbl_real = torch.ones(batch_size, 1, device=device)
  #print('lbl_real.size() =', lbl_real.size())
  lbl_fake = torch.zeros(batch_size, 1, device=device)

  fixed_noise = torch.randn(100, G.noise_dim, device=device) #for plotting

  start_time = time.perf_counter()

  for epoch in range(curves['epoch'], epochs):

    for x_real, _ in dataloader:

      curves['batch'] += 1

      # STEP 1: D optimization step
      x_real = x_real.to(device)
      z = torch.randn(batch_size, G.noise_dim, device=device)
      with torch.no_grad():
        x_gen = G(z)
      optimizerD.zero_grad()
      lossD = get_disciminator_loss(D, x_real, x_gen, lbl_real, lbl_fake)
      lossD.backward()
      if grad_max_norm is not None:
        nn.utils.clip_grad_norm_(D.parameters(), grad_max_norm)
      optimizerD.step()

      # STEP 2: G optimization step
      z = torch.randn(batch_size, G.noise_dim, device=device)
      optimizerG.zero_grad()
      lossG = get_generator_loss(G, D, z, lbl_real) # we use the unrolled D
      lossG.backward()
      if grad_max_norm is not None:
        nn.utils.clip_grad_norm_(G.parameters(), grad_max_norm)
      optimizerG.step()

      if eval_avg:
        update_avg_gen(G, G_avg, curves['batch'])
        update_ema_gen(G, G_ema, beta_ema=0.9999)

    # Just plotting things
    with torch.no_grad():
      probas = torch.sigmoid(D(G(fixed_noise)))
      mean_proba = probas.mean().cpu().item()
      std_proba = probas.std().cpu().item()
      samples = G(fixed_noise)
    print(f"Epoch {epoch}: Mean proba from D(G(z)): {mean_proba:.4f} +/- {std_proba:.4f}")
    plot_func(samples.detach().cpu(), time_tick=time.perf_counter() - start_time, D=D, G=G, epoch=epoch+1, G_avg=G_avg, G_ema=G_ema, curves=curves)

    save_models(G, D, G_avg, G_ema, optimizerG, optimizerD, current_dir, suffix="last")
    save_curves(curves, current_dir)


def save_curves(curves, out_dir):
  curves_file_name = os.path.join(out_dir, 'curves.json')
  with open(curves_file_name, 'w') as fs:
    json.dump(curves, fs)


def load_curves(out_dir):
  curves_file_name = os.path.join(out_dir, 'curves.json')
  with open(curves_file_name, 'r') as fs:
    curves = json.load(fs)
  return curves


def get_optimizer(model, args):
  
  optim_name = args['optim']
  lr = args['lr']
  betas = args['betas']
  lookahead = args['lookahead']
  lookahead_k = args['lookahead_k']

  if optim_name == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
  elif optim_name == 'ExtraAdam':
    optimizer = ExtraAdam(model.parameters(), lr=lr, betas=betas)
  else:
    raise ValueError('Unexpected value for optimizer')
  
  if lookahead:
    optimizer = Lookahead(optimizer, k=lookahead_k)
  
  return optimizer


def get_dir(args, out_dir):
  version = f"optim{args['optim']}_lr{args['lr']}_bs{args['batch_size']}_" + \
            f"lookahead{args['lookahead']}_lak{args['lookahead_k']}_seed{args['seed']}"
  current_dir = os.path.join(out_dir, version)
  return current_dir



def save_models(G, D, G_avg, G_ema, opt_G, opt_D, out_dir, suffix):

  torch.save(G.state_dict(), os.path.join(out_dir, f"gen_{suffix}.pth"))
  torch.save(D.state_dict(), os.path.join(out_dir, f"disc_{suffix}.pth"))
  if G_avg is not None:
    torch.save(G_avg.state_dict(), os.path.join(out_dir, f"gen_avg_{suffix}.pth"))
  if G_ema is not None:
    torch.save(G_ema.state_dict(), os.path.join(out_dir, f"gen_ema_{suffix}.pth"))
  torch.save(opt_G.state_dict(), os.path.join(out_dir, f"gen_optim_{suffix}.pth"))
  torch.save(opt_D.state_dict(), os.path.join(out_dir, f"disc_optim_{suffix}.pth"))


def load_models(G, D, G_avg, G_ema, opt_G, opt_D, out_dir, suffix):
  
  G.load_state_dict(torch.load(os.path.join(out_dir, f"gen_{suffix}.pth")))
  D.load_state_dict(torch.load(os.path.join(out_dir, f"disc_{suffix}.pth")))
  if G_avg is not None:
    G_avg.load_state_dict(torch.load(os.path.join(out_dir, f"gen_avg_{suffix}.pth")))
  if G_ema is not None:
    G_ema.load_state_dict(torch.load(os.path.join(out_dir, f"gen_ema_{suffix}.pth")))
  opt_G.load_state_dict(torch.load(os.path.join(out_dir, f"gen_optim_{suffix}.pth")))
  opt_D.load_state_dict(torch.load(os.path.join(out_dir, f"disc_optim_{suffix}.pth")))