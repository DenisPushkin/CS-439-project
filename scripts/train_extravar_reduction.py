import shutil, fire, os, copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

import optimizers
from train import get_discriminator_loss, get_generator_loss, save_models
from dataloader import load_mnist
from plot import get_plot_func
from utils import elapsed
from models import GeneratorCNN28, DiscriminatorCNN28


def sampler(data, batch_size):
    if len(data) % batch_size is not 0:
        raise ValueError("Batch size should be divisible by mini batch size")
    start = 0
    while True:
        yield data[start:start+batch_size]
        start += batch_size
        start %= len(data)


def train(epochs=5, batch_size=1024, 
          mini_batch_size=32, p=None, lr=1e-4,
          betas=(0.5, 0.999), alpha=0.9, eval_every=100, 
          n_workers=4, device=torch.device('cuda'),
          grad_max_norm=1, plot_func=None,
          out_dir=None, shuffle=True, 
          pretrained_clf_path="./mnist.pth", seed=None
):
    if p is None:
        p = 2*mini_batch_size/batch_size
    version = "epochs{}_bs{}_mbs{}_lr{}_betas{}_alpha{}_p{}_ee{}_seed{}".format(
        epochs, batch_size, mini_batch_size, lr, betas, alpha, p, eval_every, seed
    )
    group_name = "Extragradient with Variance Reduction"
    logger = wandb.init(project="CS-439", entity="y-kivva", group=group_name, reinit=True)
    wandb_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "mini_batch_size": mini_batch_size,
        "lr": lr,
        "betas": betas,
        "alpha": alpha,
        "seed": seed,
        "p": p
    }
    logger.config.update(wandb_config)
    logger.name = f"BS:{batch_size}, mBS:{mini_batch_size}, seed:{seed}, lr:{lr}"
    
    current_dir = os.path.join(out_dir, version)
    shutil.rmtree(current_dir, ignore_errors=True)
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
        
    dataset = load_mnist(_data_root='datasets', binarized=False)
    if plot_func is None:
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
    
    optimizer_GAN = optimizers.Extragrad_Var_Reduction(
        [
            {"params": D.parameters()},
            {"params": G.parameters()}
        ],
        [
            {"params": dual_D.parameters()},
            {"params": dual_G.parameters()}
        ],
        lr=lr,
        betas=betas,
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
    for epoch in range(epochs):
        i = 0
        for data, _ in data_sampler:
            i += 1
            x_real = data.to(device)
            z = torch.randn(batch_size, G.noise_dim, device=device)
            
            optimizer_GAN.zero_grad()
            lossG = get_generator_loss(dual_G, dual_D, z, lbl_real)
            lossG.backward(inputs=list(dual_G.parameters()))
            if grad_max_norm is not None:
                nn.utils.clip_grad_norm_(dual_G.parameters(), grad_max_norm)
            
            with torch.no_grad():
                x_gen = dual_G(z)
            lossD = get_discriminator_loss(dual_D, x_real, x_gen, lbl_real, lbl_fake)
            lossD.backward(inputs=list(dual_D.parameters()))
            if grad_max_norm is not None:
                nn.utils.clip_grad_norm_(dual_D.parameters(), grad_max_norm)
            
            mini_data_iter  = sampler(x_real, mini_batch_size)
            def closure():
                x_real = next(mini_data_iter)
                z = torch.randn(mini_batch_size, G.noise_dim, device=device)
                
                optimizer_GAN.zero_grad()
                dual_lossG = get_generator_loss(dual_G, dual_D, z, lbl_real_mini)
                lossG = get_generator_loss(G, D, z, lbl_real_mini)
                lossG.backward(inputs=list(G.parameters()))
                dual_lossG.backward(inputs=list(dual_G.parameters()))
                if grad_max_norm is not None:
                    nn.utils.clip_grad_norm_(dual_G.parameters(), grad_max_norm)
                    nn.utils.clip_grad_norm_(G.parameters(), grad_max_norm)
                
                with torch.no_grad():
                    dual_x_gen = dual_G(z)
                    x_gen = G(z)
                
                dual_lossD = get_discriminator_loss(dual_D, x_real, dual_x_gen,
                                                    lbl_real_mini, lbl_fake_mini)
                lossD = get_discriminator_loss(D, x_real, x_gen, lbl_real_mini, lbl_fake_mini)
                lossD.backward(inputs=list(D.parameters()))
                dual_lossD.backward(inputs=list(dual_D.parameters()))
                return (lossG, lossD), (dual_lossG, dual_lossD)
            
            loss, dual_loss = optimizer_GAN.step(closure)
            lossG, lossD = loss
            dual_lossG, dual_lossD = dual_loss
            
            working_time += elapsed()
            if i % eval_every == 0:
                logger.log({
                    "Working time": working_time,
                    "Generator Loss": lossG,
                    "Dual Generator Loss": dual_lossG,
                    "Discriminator Loss": lossD,
                    "Dual Discriminator Loss": dual_lossD
                })
                if out_dir is not None:
                    save_models(G, D, out_dir=out_dir, suffix="last")
                with torch.no_grad():
                    probas = torch.sigmoid(D(G(fixed_noise)))
                    mean_proba = probas.mean().cpu().item()
                    std_proba = probas.std().cpu().item()
                    samples = G(fixed_noise)
                    print(f"Iter {i}: Mean proba from D(G(z)): {mean_proba:.4f} +/- {std_proba:.4f}")
                    plot_func(samples.detach().cpu(), time_tick=working_time, 
                              D_loss=lossD.item(), G_loss=lossG.item(),
                              D=D, G=G, iteration=i)
    logger.finish()
                    
                
if __name__=="__main__":
    fire.Fire(train)
