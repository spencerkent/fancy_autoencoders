"""
Some plotting functionality for analyzing the fancy autoencoders
"""

import bisect
import numpy as np
from scipy.stats import kurtosis
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

def plot_weights(weight_tensor, img_params, enc_dec='encoding',
                 renormalize=True, single_img=False):
  """
  Plot each of the weight vectors reshaped into image space

  Parameters
  ----------
  weight_tensor : ndarray
      In the case of encoding weights, the D x H matrix of weight vectors, where
      D is the flattened dimensionality of each image and H is the number of
      hidden units (code length). Each weight vector is a column in this matrix.
      In the case of decoding weights, this is H x D where decoding weight is
      one of the rows of the matrix
  img_params : dictionary
      'height' and 'width'
  enc_dec : str
      'encoding' or 'decoding'
  renormalize : bool, optional
      If true, take each weight vector and renormalize to the interval [-1, 1].
      Do this for each weight vector independently
  single_img : bool, optional
      If true, just make a single composite image with all the weight vectors
      separated by some margin in X and Y. A lot faster than many different
      suplots but you lose control and the ability to label things individually.
      Default False.

  Returns
  -------
  weight_figs : list
      A list containing pyplot figures. Can be saved separately, or whatever
      from the calling function
  """
  if enc_dec == 'decoding':
    weight_tensor = weight_tensor.T  # we want to see the *output* weights

  max_weight_val = np.max(weight_tensor)
  min_weight_val = np.min(weight_tensor)

  max_weights_per_fig = 400  # for displaying individual filters
  assert np.sqrt(max_weights_per_fig) % 1 == 0, 'please pick a square number'
  num_weights = weight_tensor.shape[1]
  num_weight_figs = int(np.ceil(num_weights / max_weights_per_fig))
  # this determines how many weights are aranged in a square grid within
  # any given figure
  if num_weight_figs > 1:
    weights_per_fig = max_weights_per_fig
  else:
    squares = [x**2 for x in range(1, int(np.sqrt(max_weights_per_fig))+1)]
    weights_per_fig = squares[bisect.bisect_left(squares, num_weights)]
  plot_sidelength = int(np.sqrt(weights_per_fig))

  if single_img:
    h_margin = 2
    w_margin = 2
    full_img_height = (img_params['height'] * plot_sidelength +
                       (plot_sidelength - 1) * h_margin)
    full_img_width = (img_params['width'] * plot_sidelength +
                      (plot_sidelength - 1) * w_margin)

  weight_idx = 0
  weight_figs = []
  for in_weight_fig_idx in range(num_weight_figs):
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(enc_dec + ' filters, fig {} of {}'.format(
                      in_weight_fig_idx+1, num_weight_figs), fontsize=15)
    if single_img:
      if renormalize:
        composite_img = np.ones((full_img_height, full_img_width))
      else:
        composite_img = (max_weight_val *
                         np.ones((full_img_height, full_img_width)))
    else:
      # each weight vector gets its own subplot
      subplot_grid = gridspec.GridSpec(plot_sidelength, plot_sidelength,
                                       wspace=0.05, hspace=0.05)
    fig_weight_idx = weight_idx % weights_per_fig
    while fig_weight_idx < weights_per_fig and weight_idx < num_weights:

      if single_img:
        if renormalize:
          this_weight = (weight_tensor[:, weight_idx] /
                         np.max(np.abs(weight_tensor[:, weight_idx])))
          #^ now guaranteed to be in [-1, 1]
        else:
          this_weight = np.copy(weight_tensor[:, weight_idx])

        # okay, now actually plot the weights in this figure
        row_idx = fig_weight_idx // plot_sidelength
        col_idx = fig_weight_idx % plot_sidelength
        pxr1 = row_idx * (img_params['height'] + h_margin)
        pxr2 = pxr1 + img_params['height']
        pxc1 = col_idx * (img_params['width'] + w_margin)
        pxc2 = pxc1 + img_params['width']
        composite_img[pxr1:pxr2, pxc1:pxc2] = \
          np.reshape(this_weight, (img_params['height'], img_params['width']))

        fig_weight_idx += 1
        weight_idx += 1

      else:
        if weight_idx % 100 == 0:
          print("plotted ", weight_idx, " of ", num_weights, " weights")
        if not renormalize:
          raise NotImplementedError('coming soon...')
        ax = plt.Subplot(fig, subplot_grid[fig_weight_idx])
        ax.imshow(np.reshape(weight_tensor[:, weight_idx],
                             (img_params['height'], img_params['width'])),
                  cmap='gray', interpolation='nearest')
        ax.axis('off')
        fig.add_subplot(ax)
        fig_weight_idx += 1
        weight_idx += 1

    if single_img:
      plt.imshow(composite_img, cmap='gray')
      plt.axis('off')

    weight_figs.append(fig)

  return weight_figs


def analyze_code(code_vectors, specific_imgs=None):
  """
  Generates some plots useful for analyzing out codes

  Parameters
  ----------
  code_vectors : ndarray
      The B x H array of codes
  specific_imgs : ndarray, optional
      If provided, show the code vectors for these specific images. Default None

  Returns
  -------
  activation_hists : pyplot.figure
      The histogram of activation values across the entire set of inputs
  code_plot : pyplot.figure
      A sampling of image codes
  """
  all_l1 = np.sum(np.abs(code_vectors), axis=1)
  mean_l1 = np.mean(all_l1)
  l0_tol = 1e-5
  all_l0 = np.sum(code_vectors > l0_tol, axis=1)
  mean_l0 = np.mean(all_l0)

  # compute avg L1, avg L0, and kurtosis

  counts, bin_edges = np.histogram(code_vectors, bins=100)
  counts = counts / np.sum(counts)
  activation_hists = plt.figure(figsize=(15, 5))
  ax = plt.subplot(1, 1, 1)
  ax.bar(bin_edges[:-1], counts, align='edge', log=True)
  ax.set_title('Histogram across whole dataset of coefficient values')
  ax.set_ylabel('Normalized count')
  ax.set_xlabel('Coefficient value')
  hist_kurtosis = kurtosis(counts, fisher=False)  # gaussian will be 3.0
  ax.text(0.01, 0.11, 'Mean L1 sparsity: ' + '{:.2f}'.format(mean_l1),
          horizontalalignment='left', verticalalignment='bottom',
          transform=ax.transAxes, color='orange', fontsize=9)
  ax.text(0.01, 0.06, 'Mean L0 sparsity: ' + '{:.2f}'.format(mean_l0) +
                      ' (tol=' + '{:.1e}'.format(l0_tol) + ')',
          horizontalalignment='left', verticalalignment='bottom',
          transform=ax.transAxes, color='orange', fontsize=9)
  ax.text(0.01, 0.01, 'Histogram kurtosis: ' + '{:.2f}'.format(hist_kurtosis),
          horizontalalignment='left', verticalalignment='bottom',
          transform=ax.transAxes, color='orange', fontsize=9)


  if specific_imgs is not None:
    assert np.sqrt(len(specific_imgs)) % 1 == 0, 'please pick a square number'
    num_samps = len(specific_imgs)
    inds = specific_imgs.copy()
  else:
    num_samps = 9
    assert np.sqrt(num_samps) % 1 == 0, 'please pick a square number'
    inds = np.random.choice(np.arange(code_vectors.shape[0]), num_samps,
                            replace=False)

  samp_codes = plt.figure(figsize=(15, 5))
  for idx in range(num_samps):
    ax = plt.subplot(int(np.sqrt(num_samps)), int(np.sqrt(num_samps)), idx + 1)
    # I really want to do stem plots but plt.stem is really slow...
    ax.plot(code_vectors[inds[idx], :])
    ax.text(0.01, 0.97, 'L1 norm: ' + '{:.2f}'.format(all_l1[inds[idx]]),
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='orange', fontsize=8)
    ax.text(0.01, 0.87, 'L0 norm: ' + '{:.2f}'.format(all_l0[inds[idx]]) +
                        ' (tol=' + '{:.1e}'.format(l0_tol) + ')',
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='orange', fontsize=8)
    if idx != num_samps - int(np.sqrt(num_samps)):
      ax.get_xaxis().set_visible(False)

  plt.suptitle('Example code vectors')

  return activation_hists, samp_codes


def plot_reconstructions(ground_truth, recons, img_params, specific_imgs=None):
  """
  Plot reconstructions generated by the model

  Parameters
  ----------
  ground_truth : ndarray
      The set of images we would like to compare against
  recons : ndarray
      The set of images the model produced
  img_params : dictionary
      'height' and 'width'
  specific_images : ndarray, optional
      If provided, we show these specific images and recons. Default None

  Returns
  -------
  gt_plot : pyplot.figure
  recon_plot : pyplot.figure
  """
  if specific_imgs is not None:
    assert np.sqrt(len(specific_imgs)) % 1 == 0, 'please pick a square number'
    num_samps = len(specific_imgs)
    inds = specific_imgs.copy()
  else:
    num_samps = 9
    assert np.sqrt(num_samps) % 1 == 0, 'please pick a square number'
    inds = np.random.choice(np.arange(code_vectors.shape[0]), num_samps,
                                 replace=False)

  gt_plot = plt.figure(figsize=(10, 10))
  for idx in range(num_samps):
    plt.subplot(int(np.sqrt(num_samps)), int(np.sqrt(num_samps)), idx + 1)
    plt.imshow(np.reshape(ground_truth[inds[idx], :],
                          (img_params['height'], img_params['width'])),
               cmap='gray')
    plt.axis('off')
  plt.suptitle('Input images')

  recon_plot = plt.figure(figsize=(10, 10))
  for idx in range(num_samps):
    # compute the PSNR of the reconstruction
    mse = np.mean(np.square(recons[inds[idx], :] - ground_truth[inds[idx], :]))
    psnr = 10 * np.log10(1. / mse)
    ax = plt.subplot(int(np.sqrt(num_samps)), int(np.sqrt(num_samps)), idx + 1)
    ax.imshow(np.reshape(recons[inds[idx], :],
                         (img_params['height'], img_params['width'])),
              cmap='gray')
    ax.text(0.01, 0.97, '{:.2f} dB'.format(psnr),
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='white', fontsize=10)
    plt.axis('off')
  plt.suptitle('Reconstructions images')

  return gt_plot, recon_plot
