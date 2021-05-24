import numpy
import os
import torchvision.utils as vision_utils
import numpy as np
import matplotlib.pyplot as plt
from models import pretrained_mnist_model
from metrics import compute_mu_sigma_pretrained_model, get_metrics


def get_plot_func(dataset, pretrained_clf_path, out_dir, img_size, num_samples_eval=10000): #save_curves
  #dataset = load_mnist(_data_root='datasets', binarized=False)
  pretrained_clf = pretrained_mnist_model(pretrained=pretrained_clf_path)
  mu_real, sigma_real = compute_mu_sigma_pretrained_model(dataset, pretrained_clf)
  #inception_means, inception_stds, inception_means_ema, inception_means_avg, fids, fids_ema, fids_avg = [], [], [], [], [], [], []
  #iterations, times = [], []
  def plot_func(samples, epoch, time_tick, curves, G=None, D=None, G_avg=None, G_ema=None):
    fig = plt.figure(figsize=(12,5), dpi=100)
    plt.subplot(1,2,1)
    samples = samples.view(100, *img_size)
    file_name = os.path.join(out_dir, '%05d.png' % epoch)
    vision_utils.save_image(samples, file_name, nrow=10)
    grid_img = vision_utils.make_grid(samples, nrow=10, normalize=True, padding=0)
    plt.imshow(grid_img.permute(1, 2, 0), interpolation='nearest')
    plt.subplot(1,2,2)
    metrics = get_metrics(pretrained_clf, num_samples_eval, mu_real, sigma_real, G)
    curves['fids'].append(metrics['fid'])
    curves['inception_means'].append(metrics['inception_mean'])
    curves['inception_stds'].append(metrics['inception_std'])
    if G_avg is not None:
      metrics = get_metrics(pretrained_clf, num_samples_eval, mu_real, sigma_real, G_avg)
      curves['fids_avg'].append(metrics['fid'])
      curves['inception_means_avg'].append(metrics['inception_mean'])
    if G_ema is not None:
      metrics = get_metrics(pretrained_clf, num_samples_eval, mu_real, sigma_real, G_ema)
      curves['fids_ema'].append(metrics['fid'])
      curves['inception_means_ema'].append(metrics['inception_mean'])
    curves['epoch'] = epoch
    curves['times'].append(time_tick)

    #  is
    epochs = np.arange(curves['epoch'])
    is_low  = [m - s for m, s in zip(curves['inception_means'], curves['inception_stds'])]
    is_high = [m + s for m, s in zip(curves['inception_means'], curves['inception_stds'])]
    plt.plot(epochs, curves['inception_means'], label="is", color='r')
    plt.fill_between(epochs, is_low, is_high, facecolor='r', alpha=.3)
    plt.yticks(np.arange(0, 10+1, 0.5))

    # fid
    plt.plot(epochs, curves['fids'], label="fid", color='b')
    plt.xlabel('Time (sec)')
    plt.ylabel('Metric')
    plt.grid()
    ax = fig.gca()
    ax.set_ylim(-0.1, 10)
    plt.legend(fancybox=True, framealpha=.5)

    #saving plots
    curves_img_file_name = os.path.join(out_dir, 'curves.png')
    fig.savefig(curves_img_file_name)
    plt.show()
    plt.close(fig)
    #curves_file_name = os.path.join(out_dir, 'curves.json')
    #with open(curves_file_name, 'w') as fs:
    #  json.dump(curves, fs)
  return plot_func