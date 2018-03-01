"""
Let's attempt to learn a code for natural image patches using a GDN autoencoder
"""
import sys
dsc_folder = '/home/spencerkent/Projects/DeepSparseCoding/'
sys.path.insert(1, dsc_folder)

import pickle
import numpy as np
from matplotlib import pyplot as plt

from models import GDN_autoencoder
# from models import RELU_autoencoder
# from models import sigmoid_autoencoder
import utils.plotting

run_identifier = 'testing_before_push'
logfile_directory = '/media/expansion1/spencerkent/logfiles/fancy_autoencoder/'

image_params = {'height': 16, 'width': 16, 'depth': 1}
hidden_layer_params = {'weight_init_std': 0.1, 'code_size': 256}
output_layer_params = {'weight_init_std': 0.1,
                       'loss_type': 'l2recon_l1sparsity_wd'}
optimization_step_schedule = {'OPT_step': {0: 0.01, 200000: 0.004,
                                           500000: 0.001},
                              'sparsity_wt': {0: 0.000},
                              'weight_decay_wt': {0: 0.13}}
batch_size = 256
# #sophias dataset for cosyne
training_set_filename = '/home/spencerkent/Datasets/vanHaterenNaturalScenes/sophias_patches_preZCAwhite_train.npz'
testing_set_filename = '/home/spencerkent/Datasets/vanHaterenNaturalScenes/test_zca_wht_v2.npz'
# baboon_filename = '/home/spencerkent/Datasets/vanHaterenNaturalScenes/baboon_zca_wht.npz'



def main():
  model_params = {'image_params': image_params,
                  'hidden_layer_params': hidden_layer_params,
                  'output_layer_params': output_layer_params}
  pickle.dump(model_params, open(logfile_directory + run_identifier +
                                 '_model_params.p', 'wb'))

  # wrangling the natural image patches dataset
  with np.load(training_set_filename, fix_imports=True) as d:
    sanborn_dset = d['arr_0'].item()['train']
  with np.load(testing_set_filename, fix_imports=True) as d:
    sanborn_dset_test = d['arr_0'].item()['train']
  # with np.load(baboon_filename, fix_imports=True) as d:
  #   baboon = d['arr_0'].item()['train']

  preproc_subtractive_term = np.min(sanborn_dset.images)
  preproc_divisive_term = np.max(sanborn_dset.images -
                                 preproc_subtractive_term)

  num_val = int(0.2 * sanborn_dset.num_examples)  # number of validation patches
  random_idx = np.arange(sanborn_dset.num_examples)
  np.random.shuffle(random_idx)
  train_dset = sanborn_dset.images[random_idx[0:-num_val], :]
  train_dset = (train_dset - preproc_subtractive_term) / preproc_divisive_term
  val_dset = sanborn_dset.images[random_idx[-num_val:], :]
  val_dset = (val_dset - preproc_subtractive_term) / preproc_divisive_term
  test_dset = np.copy(sanborn_dset_test.images)
  test_dset = (test_dset - preproc_subtractive_term) / preproc_divisive_term

  # now some mean subtraction
  raw_component_means = np.mean(train_dset, axis=0)
  train_dset = train_dset - raw_component_means
  val_dset = val_dset - raw_component_means
  test_dset = test_dset - raw_component_means
  np.savez(open(
    '/home/spencerkent/Datasets/vanHaterenNaturalScenes/preproc_params.npz',
    'wb'), means=raw_component_means, preproc_div=preproc_divisive_term,
           preproc_sub=preproc_subtractive_term)




  print("Constructing Tensorflow Graph")
  sig_ae = GDN_autoencoder(hidden_layer_params, output_layer_params,
                                    image_params)

  # training from scratch
  tfsession = sig_ae.StartSession()

  trn_report = sig_ae.Train(tfsession, train_dset, val_dset, batch_size,
                            optimization_step_schedule, snapshot_progress=True,
                            logging_dir=logfile_directory, report_every=1000,
                            max_epochs=150)

  sig_ae.SaveSession(tfsession, logfile_directory, run_identifier)

  # plot the learned encoding and decoding weights
  learned_enc, learned_dec = sig_ae.GetEncDecWeights(tfsession)
  enc_figs = utils.plotting.plot_weights(learned_enc, image_params, 'encoding',
                                         single_img=True, renormalize=True)
  dec_figs = utils.plotting.plot_weights(learned_dec, image_params, 'decoding',
                                         single_img=True, renormalize=True)
  for idx in range(len(enc_figs)):
    enc_figs[idx].savefig(logfile_directory + run_identifier +
                          '_enc_weights' + str(idx) + '.png')
    plt.close(enc_figs[idx])
  for idx in range(len(dec_figs)):
    dec_figs[idx].savefig(logfile_directory + run_identifier +
                          '_dec_weights' + str(idx) + '.png')
    plt.close(dec_figs[idx])

  some_rand_inds = np.random.choice(np.arange(val_dset.shape[0]), 9,
                                    replace=False)
  coeffs = sig_ae.GetCodes(tfsession, val_dset)
  coeff_figs = utils.plotting.analyze_code(coeffs, some_rand_inds)
  coeff_figs[0].savefig(logfile_directory + run_identifier +
                        '_activation_hists.png')
  plt.close(coeff_figs[0])
  coeff_figs[1].savefig(logfile_directory + run_identifier +
                        '_random_samp_codes.png')
  plt.close(coeff_figs[1])
  final_loss, final_recons = sig_ae.Test(tfsession, val_dset, True)
  recon_figs = utils.plotting.plot_reconstructions(val_dset, final_recons,
      image_params, some_rand_inds)
  recon_figs[0].savefig(logfile_directory + run_identifier +
                        '_samp_gt_imgs.png')
  plt.close(recon_figs[0])
  recon_figs[1].savefig(logfile_directory + run_identifier +
                        '_samp_recon_imgs.png')
  plt.close(recon_figs[1])

  tfsession.close()

if __name__ == '__main__':
  main()

