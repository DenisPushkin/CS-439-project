import shutil, fire, os, copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

import optimizers
from metrics import get_discriminator_loss, get_generator_loss
from plot import get_plot_func
from utils import elapsed, load_mnist, sampler
from models import GeneratorCNN28, DiscriminatorCNN28, save_models


def train(epochs=5, batch_size=1024, 
          mini_batch_size=64, p=None, lr=1e-4,
          betas=(0.5, 0.999), alpha=None, eval_every=100, 
          n_workers=4, device=torch.device('cuda'),
          grad_max_norm=1, out_dir=None, shuffle=True, 
          pretrained_clf_path="./mnist.pth", seed=None
):
    if p is None:
        p = 2*mini_batch_size/batch_size
    if alpha is None:
        alpha = 1 - p
    version = "epochs{}_bs{}_mbs{}_lr{}_betas{}_alpha{}_p{}_ee{}_seed{}".format(
        epochs, batch_size, mini_batch_size, lr, betas, alpha, p, eval_every, seed
    )
    current_dir = os.path.join(out_dir, version)
    shutil.rmtree(current_dir, ignore_errors=True)
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
        
    dataset = load_mnist(_data_root='datasets')
    plot_func = get_plot_func(
        dataset,
        out_dir=current_dir,
        pretrained_clf_path=pretrained_clf_path,
        img_size=dataset[0][0].size(),
        num_samples_eval=10000
    )
    
    if seed is not None:
        torch.manual_seed(seed)
        
    dataloader = DataLoader(dataset, batch_size, 
                            shuffle=shuffle, 
                            num_workers=n_workers,
                            drop_last=True)
    data_sampler = iter(dataloader)
        
    G = GeneratorCNN28()
    D = DiscriminatorCNN28(img_size=28)
    
    optimizer = torch.optim.Adam
    
    dual_D = copy.deepcopy(D)
    dual_G = copy.deepcopy(G)
    
    optimizer_min_max = optimizers.Extragrad_Var_Reduction_Original(
        parameters=[
            {
                "params":D.parameters(),
            },
            {
                "params":G.parameters(),
            }
        ],
        dual_parameters=[
            {
                "params":dual_D.parameters(),
            },
            {
                "params":dual_G.parameters(),
            }
        ],
        lr=lr,
        alpha=alpha,
        p=p,
        optimizer=optimizer
    )
    
    #LBLs
    lbl_real = torch.ones(batch_size, 1, device=device)
    lbl_fake = torch.zeros(batch_size, 1, device=device)
    lbl_real_mini = torch.ones(mini_batch_size, 1, device=device)
    lbl_fake_mini = torch.zeros(mini_batch_size, 1, device=device)
    
    fixed_noise = torch.randn(100, G.noise_dim, device=device)
    
    G.to(device)
    D.to(device)
    
    dual_D.to(device)
    dual_G.to(device)
    
    
    working_time = 0
    i = 0
    for epoch in range(epochs):
        data_sampler = iter(dataloader)
        for data, _ in data_sampler:
            i += 1
            x_real = data.to(device)

            optimizer_min_max.zero_grad()
            
            # Get grads with respect to the Generator
            z = torch.randn(batch_size, G.noise_dim, device=device)
            dual_lossG = get_generator_loss(dual_G, dual_D, z, lbl_real)
            lossG = get_generator_loss(G, D, z, lbl_real)
            dual_lossG.backward(inputs=list(dual_G.parameters()))
            lossG.backward(inputs=list(G.parameters()))
            if grad_max_norm is not None:
                nn.utils.clip_grad_norm_(dual_G.parameters(), grad_max_norm)
                nn.utils.clip_grad_norm_(G.parameters(), grad_max_norm)
            
            # Get grads with respect to the Discriminator
            with torch.no_grad():
                x_gen = G(z)
                dual_x_gen = dual_G(z)
            
            x_real = data.to(device)
            lossD = get_discriminator_loss(D, x_real, x_gen, lbl_real, lbl_fake)
            dual_lossD = get_discriminator_loss(dual_D, x_real, dual_x_gen, lbl_real, lbl_fake)
            
            lossD.backward()
            dual_lossD.backward()
            if grad_max_norm is not None:
                nn.utils.clip_grad_norm_(D.parameters(), grad_max_norm)
                nn.utils.clip_grad_norm_(dual_D.parameters(), grad_max_norm)
            
            z_mini_iter = sampler(z, mini_batch_size)
            mini_data_iter  = sampler(x_real, mini_batch_size)
            def closure():
                z_mini = next(z_mini_iter)
                x_real_mini = next(mini_data_iter)
                lossG_mini = get_generator_loss(G, D, z_mini, lbl_real_mini)
                dual_lossG_mini = get_generator_loss(dual_G, dual_D, z_mini, lbl_real_mini)
                lossG_mini.backward(inputs=list(G.parameters()))
                dual_lossG_mini.backward(inputs=list(dual_G.parameters()))
                if grad_max_norm is not None:
                    nn.utils.clip_grad_norm_(G.parameters(), grad_max_norm)
                    nn.utils.clip_grad_norm_(dual_G.parameters(), grad_max_norm)
                    
                with torch.no_grad():
                    x_gen_mini = G(z_mini)
                    dual_x_gen_mini = dual_G(z_mini)
                
                lossD_mini = get_discriminator_loss(
                    D, x_real_mini, x_gen_mini, lbl_real_mini, lbl_fake_mini
                )
                dual_lossD_mini = get_discriminator_loss(
                    dual_D, x_real_mini, dual_x_gen_mini, lbl_real_mini, lbl_fake_mini
                )
                lossD_mini.backward()
                dual_lossD_mini.backward()
                return (lossG_mini, dual_lossG_mini), (lossD_mini, dual_lossD_mini)
            
            (lossG, dual_lossG), (lossD, dual_lossD)  = optimizer_min_max.step(closure)
            dual_D.load_state_dict(D.state_dict())
            dual_G.load_state_dict(G.state_dict())
            
            working_time += elapsed()
            if i % eval_every == 0:
                if out_dir is not None:
                    save_models(G, D, out_dir=out_dir, suffix="last")
                with torch.no_grad():
                    probas = torch.sigmoid(D(G(fixed_noise)))
                    mean_proba = probas.mean().cpu().item()
                    std_proba = probas.std().cpu().item()
                    samples = G(fixed_noise)
                    print(f"Epoch:{epoch} Iter {i}: Mean proba from D(G(z)): {mean_proba:.4f} +/- {std_proba:.4f}")
                    plot_func(samples.detach().cpu(), time_tick=working_time, 
                              D_loss=lossD.item(), G_loss=lossG.item(),
                              D=D, G=G, iteration=i)
                    
                
if __name__=="__main__":
    fire.Fire(train)
