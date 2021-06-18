import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import copy

from .utils import elapsed, get_sampler
from .lookahead import Lookahead
from .models import GeneratorCNN28, DiscriminatorCNN28


def get_discriminator_loss(D, x_real, x_gen, lbl_real, lbl_fake):
  """"""
  D_x = D(x_real)
  D_G_z = D(x_gen)
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


def train(dataset, iterations, batch_size=32, lr=1e-4,
          betas=(0.5, 0.999), eval_every=100, n_workers=5, device=torch.device('cuda'), 
          grad_max_norm=1, plot_func=lambda a,b,c,d: None, extragrad=False, 
          lookahead=False, lookahead_k=5, eval_avg=False, out_dir=None, seed=None):
  
  if seed is not None:
    torch.manual_seed(seed)
  
  sampler = get_sampler(dataset, batch_size, shuffle=True, 
                        num_workers=n_workers, drop_last=True)
  
  G = GeneratorCNN28()
  D = DiscriminatorCNN28(img_size=28)

  if extragrad:
    D_extra = copy.deepcopy(D)
    G_extra = copy.deepcopy(G)
  else:
    D_extra = D
    G_extra = G

  # Optimizers
  optimizerD = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
  optimizerG = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
  if lookahead:
    optimizerD = Lookahead(optimizerD, k=lookahead_k)
    optimizerG = Lookahead(optimizerG, k=lookahead_k)

  optimizerD_extra = torch.optim.Adam(D_extra.parameters(), lr=lr, betas=betas)
  optimizerG_extra = torch.optim.Adam(G_extra.parameters(), lr=lr, betas=betas)

  # LBLs
  lbl_real = torch.ones( batch_size, 1, device=device)
  lbl_fake = torch.zeros(batch_size, 1, device=device)

  fixed_noise = torch.randn(100, G.noise_dim, device=device)

  G.to(device)
  D.to(device)

  G_extra.to(device)
  D_extra.to(device)

  G_avg, G_ema = None, None
  if eval_avg:
    G_avg = copy.deepcopy(G)
    G_ema = copy.deepcopy(G)

  working_time = 0
  for i in range(iterations):

    # STEP 1: update G_extra
    elapsed()
    if extragrad:
      optimizerG_extra.zero_grad()
      z = torch.randn(batch_size, G_extra.noise_dim, device=device)
      lossG = get_generator_loss(G_extra, D, z, lbl_real)
      lossG.backward()
      optimizerG_extra.step()

    # STEP 2: update D_extra
    if extragrad:
      optimizerD_extra.zero_grad()
      x_real, _ = sampler()
      x_real = x_real.to(device)
      z = torch.randn(batch_size, G.noise_dim, device=device)
      with torch.no_grad():
        x_gen = G(z)
      lossD = get_discriminator_loss(D_extra, x_real, x_gen, lbl_real, lbl_fake)
      lossD.backward()
      optimizerD_extra.step()

    # STEP 3: D optimization step using G_extra
    x_real, _ = sampler()
    x_real = x_real.to(device)
    z = torch.randn(batch_size, G.noise_dim, device=device)
    with torch.no_grad():
      x_gen = G_extra(z) # using G_{t+1}
    optimizerD.zero_grad()
    lossD = get_discriminator_loss(D, x_real, x_gen, lbl_real, lbl_fake)
    lossD.backward()
    if grad_max_norm is not None:
      nn.utils.clip_grad_norm_(D.parameters(), grad_max_norm)
    optimizerD.step()

    # STEP 4: G optimization step using D_extra
    z = torch.randn(batch_size, G.noise_dim, device=device)
    optimizerG.zero_grad()
    lossG = get_generator_loss(G, D_extra, z, lbl_real) # we use the unrolled D
    lossG.backward()
    if grad_max_norm is not None:
      nn.utils.clip_grad_norm_(G.parameters(), grad_max_norm)
    optimizerG.step()

    if extragrad:
      G_extra.load_state_dict(G.state_dict())
      D_extra.load_state_dict(D.state_dict())

    if eval_avg:
      update_avg_gen(G, G_avg, i)
      update_ema_gen(G, G_ema, beta_ema=0.9999)

    working_time += elapsed()
    # Just plotting things
    if i % eval_every == 0 or i == iterations-1:
      if out_dir is not None:
        save_models(G, D, optimizerG, optimizerD, out_dir, suffix="last")
      with torch.no_grad():
        probas = torch.sigmoid(D(G(fixed_noise)))
        mean_proba = probas.mean().cpu().item()
        std_proba = probas.std().cpu().item()
        samples = G(fixed_noise)
      print(f"Iter {i}: Mean proba from D(G(z)): {mean_proba:.4f} +/- {std_proba:.4f}")
      plot_func(samples.detach().cpu(), time_tick=working_time, D_loss=lossD.item(), G_loss=lossG.item(), D=D, G=G, iteration=i, G_avg=G_avg, G_ema=G_ema)


def save_models(G, D, opt_G, opt_D, out_dir, suffix):
  torch.save(G.state_dict(), os.path.join(out_dir, f"gen_{suffix}.pth"))
  torch.save(D.state_dict(), os.path.join(out_dir, f"disc_{suffix}.pth"))
  if opt_G is not None:
    torch.save(opt_G.state_dict(), os.path.join(out_dir, f"gen_optim_{suffix}.pth"))
  if opt_D is not None:
    torch.save(opt_D.state_dict(), os.path.join(out_dir, f"disc_optim_{suffix}.pth"))