"""
Let's attempt to learn a code for natural image patches using a GDN autoencoder
"""
import sys
dsc_folder = '/home/spencerkent/Projects/DeepSparseCoding/'
sys.path.insert(1, dsc_folder)

import numpy as np

from models import GDN_autoencoder
from models import RELU_autoencoder
import utils.plotting

run_identifier = 'Hunting_for_filters_GDN_0_1s_15wd'

image_params = {'height': 16, 'width': 16, 'depth': 1}
hidden_layer_params = {'weight_init_std': 0.1, 'code_size': 256}  # two times overcomplete
output_layer_params = {'weight_init_std': 0.1,
                       'loss_type': 'l2_recon_l1_sparsity',
                       'init_sparsity_weight': 0.1,
                       'init_weight_decay_weight': 0.15}
optimization_step_schedule = {0: 0.001, 30000: 0.0005,
                              50000: 0.0001, 150000: 0.00002}
batch_size = 500

# #sophias dataset for cosyne
training_set_filename = '/home/spencerkent/Datasets/vanHaterenNaturalScenes/sophias_patches_preZCAwhite_train.npz'
testing_set_filename = '/home/spencerkent/Datasets/vanHaterenNaturalScenes/test_zca_wht_v2.npz'
# baboon_filename = '/home/spencerkent/Datasets/vanHaterenNaturalScenes/baboon_zca_wht.npz'
with np.load(training_set_filename, fix_imports=True) as d:
  sanborn_dset = d['arr_0'].item()['train']
with np.load(testing_set_filename, fix_imports=True) as d:
  sanborn_dset_test = d['arr_0'].item()['train']
# with np.load(baboon_filename, fix_imports=True) as d:
#   baboon = d['arr_0'].item()['train']


num_val = int(0.2 * sanborn_dset.num_examples)  # number of validation patches
random_idx = np.arange(sanborn_dset.num_examples)
np.random.shuffle(random_idx)
train_dset = sanborn_dset.images[random_idx[0:-num_val], :]
val_dset = sanborn_dset.images[random_idx[-num_val:], :]
test_dset = np.copy(sanborn_dset_test.images)
raw_component_means = np.mean(train_dset, axis=0)
#^ save this to add back in later
train_dset = train_dset - raw_component_means[None, :]
val_dset = val_dset - raw_component_means[None, :]
test_dset = test_dset - raw_component_means[None, :]


def main():

  logfile_directory = '/media/expansion1/spencerkent/logfiles/fancy_autoencoder/'

  print("Constructing Tensorflow Graph")
  gdn_autoencoder = GDN_autoencoder(hidden_layer_params, output_layer_params,
                                    image_params, optimization_step_schedule[0])

  tfsession = gdn_autoencoder.StartSession()

  trn_report = gdn_autoencoder.Train(tfsession, train_dset, val_dset,
                                     batch_size, optimization_step_schedule,
                                     snapshot_progress=True,
                                     logging_dir=logfile_directory,
                                     report_every=1000, max_epochs=200)
  gdn_autoencoder.SaveSession(tfsession, logfile_directory, run_identifier)

  # plot the learned encoding and decoding weights
  learned_enc, learned_dec = gdn_autoencoder.GetEncDecWeights(tfsession)
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
  coeffs = gdn_autoencoder.GetCodes(tfsession, val_dset)
  coeff_figs = utils.plotting.analyze_code(coeffs, some_rand_inds)
  coeff_figs[0].savefig(logfile_directory + run_identifier +
                        '_activation_hists.png')
  plt.close(coeff_figs[0])
  coeff_figs[1].savefig(logfile_directory + run_identifier +
                        '_random_samp_codes.png')
  plt.close(coeff_figs[1])
  final_loss, final_recons = gdn_autoencoder.Test(tfsession, val_dset, True)
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

