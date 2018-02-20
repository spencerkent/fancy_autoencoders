"""
Let's attempt to learn a code for natural image patches using a GDN autoencoder
"""
import sys
dsc_folder = '/home/spencerkent/Projects/DeepSparseCoding/'
sys.path.insert(1, dsc_folder)

import numpy as np

from models import GDN_autoencoder

image_params = {'height': 16, 'width': 16, 'depth': 1}
hidden_layer_params = {'weight_init_std': 0.01, 'bias_init': 0.0,
                       'code_size': 512}  # two times overcomplete
output_layer_params = {'weight_init_std': 0.01, 'bias_init': 0.0,
                       'loss_type': 'l2_recon_l1_sparsity',
                       'init_sparsity_weight': 0.1}
optimization_step_schedule = {0: 0.001, 30000: 0.0005, 50000: 0.0001}
batch_size = 500

#sophias dataset for cosyne
training_set_filename = '/home/spencerkent/Datasets/vanHaterenNaturalScenes/sophias_patches_preZCAwhite_train.npz'
testing_set_filename = '/home/spencerkent/Datasets/vanHaterenNaturalScenes/test_zca_wht_v2.npz'
# baboon_filename = '/home/spencerkent/Datasets/vanHaterenNaturalScenes/baboon_zca_wht.npz'
with np.load(training_set_filename, fix_imports=True) as d:
  sanborn_dset = d['arr_0'].item()['train']
with np.load(testing_set_filename, fix_imports=True) as d:
  sanborn_dset_test = d['arr_0'].item()['train']
# with np.load(baboon_filename, fix_imports=True) as d:
#   baboon = d['arr_0'].item()['train']


num_val = int(0.2 * sanborn_dset.num_examples)
random_idx = np.arange(sanborn_dset.num_examples)
np.random.shuffle(random_idx)
train_dset = sanborn_dset.images[random_idx[0:-num_val], :]
val_dset = sanborn_dset.images[random_idx[-num_val:], :]
test_test = np.copy(sanborn_dset_test.images)


def main():
  print("Constructing Tensorflow Graph")
  gdn_autoencoder = GDN_autoencoder(hidden_layer_params, output_layer_params,
                                    image_params, optimization_step_schedule[0])

  tfsession = gdn_autoencoder.StartSession()

  trn_report = gdn_autoencoder.Train(tfsession, train_dset, val_dset,
                                     batch_size, optimization_step_schedule,
                                     report_every=1000, max_epochs=100)

  logfile_directory = '/media/expansion1/spencerkent/logfiles/gdn_autoencoder/'
  gdn_autoencoder.SaveSession(tfsession, logfile_directory, 'testing_save_p')

  tfsession.close()

if __name__ == '__main__':
  main()

