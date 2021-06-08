import shutil
import os
import json
import torch
from dataloader import load_mnist
from plot import get_plot_func
from train import train


def get_dir(args, out_dir):
    version = f"iter{args['iterations']}_bs{args['batch_size']}_lr{args['lr']}" + \
              f"_betas{args['betas']}_lookahead{args['lookahead']}" + \
              f"_lak{args['lookahead_k']}" + \
              f"_extragrad{args['extragrad']}_ee{args['eval_every']}_seed{args['seed']}"
    current_dir = os.path.join(out_dir, version)
    return current_dir


def main(args, out_dir, pretrained_clf_path):
    
    current_dir = get_dir(args, out_dir)
    
    shutil.rmtree(current_dir, ignore_errors=True)
    if not os.path.exists(current_dir):
      os.makedirs(current_dir)
    
    with open(os.path.join(current_dir, 'args.json'), 'w') as fs:
      json.dump(args, fs)
    
    dataset = load_mnist(_data_root='datasets', binarized=False)
    
    plot_func = get_plot_func(dataset,
                              out_dir=current_dir,
                              pretrained_clf_path=pretrained_clf_path,
                              img_size=dataset[0][0].size(),
                              num_samples_eval=10000)
    
    train(dataset, 
            iterations=args['iterations'], 
            batch_size=args['batch_size'], 
            lookahead=args['lookahead'],
            lookahead_k=args['lookahead_k'],
            eval_avg=args['eval_avg'],
            lr=args['lr'], 
            betas=args['betas'], 
            extragrad=args['extragrad'],
            eval_every=args['eval_every'], 
            n_workers=args['n_workers'], 
            device=torch.device(args['device']), 
            grad_max_norm=args['grad_max_norm'], 
            seed = args['seed'],
            plot_func=plot_func,
            out_dir=current_dir)