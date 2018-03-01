"""
Demonstrates how you can use the trained model to examine the GDN code
"""
import sys
dsc_folder = 'YOUR_DSC_INSTALL_LOCATION'
sys.path.insert(1, dsc_folder)
#^ might be useful for displaying coded images

import pickle
import numpy as np
from matplotlib import pyplot as plt

from models import GDN_autoencoder
import utils.plotting

logfile_directory = 'THE_LOGFILE_DIRECTORY_I_SENT_YOU'
data_directory = 'THE_DATA_DIRECTORY_I_SENT_YOU'
run_identifier = 'training_run_for_dylan_and_sophia'

model_params = pickle.load(open(logfile_directory + run_identifier +
                                '_model_params.p', 'rb'))


image_params = model_params['image_params']
hidden_layer_params = model_params['hidden_layer_params']
output_layer_params = model_params['output_layer_params']

# Wrangling the image data
with np.load(data_directory + 'sophias_patches_preZCAwhite_train.npz',
             fix_imports=True) as d:
  sanborn_dset_train = d['arr_0'].item()['train']
with np.load(data_directory + 'sophias_patches_preZCAwhite_test_v2.npz',
             fix_imports=True) as d:
  sanborn_dset_test = d['arr_0'].item()['train']
with np.load(data_directory + 'preproc_params.npz', fix_imports=True) as d:
  preproc_means = d['means']
  preproc_subtractive_term = d['preproc_sub']
  preproc_divisive_term = d['preproc_div']

# before we load images into the net we renormalize them to the
# interval [0, 1] and then do mean subtraction.
train_dset = sanborn_dset_train.images
test_dset = sanborn_dset_test.images
train_dset = (train_dset - preproc_subtractive_term) / preproc_divisive_term
train_dset = train_dset - preproc_means
test_dset = (test_dset - preproc_subtractive_term) / preproc_divisive_term
test_dset = test_dset - preproc_means


# okay let's load in the model
gdn_ae = GDN_autoencoder(hidden_layer_params, output_layer_params,
                         image_params)
tfsession = gdn_ae.StartSession(logfile_directory + run_identifier + '.ckpt')

# we can generate plots of the encoding and decoding weights or other things,
# see functionality in utils/plotting
encoding_weights, decoding_weights = gdn_ae.GetEncDecWeights(tfsession)
encoding_weight_figs = utils.plotting.plot_weights(
    encoding_weights, image_params, single_img=True, renormalize=True)


# we can run some inputs through the net with the Test method
_, gdn_reconstructions = gdn_ae.Test(tfsession, test_dset,
                                     return_reconstructions=True)
# remember if you want to compare these against other models you'll probably
# have to put them back in the same range:
gdn_reconstructions = gdn_reconstructions + preproc_means
gdn_reconstructions = ((gdn_reconstructions * preproc_divisive_term) +
                        preproc_subtractive_term)


# we can fetch the 'GDN code for a set of inputs
gdn_codes = gdn_ae.GetCodes(tfsession, test_dset)


# we might want to fiddle with the codes a bit (quantize, etc) and then we
# can generate reconstructions from these new codes
different_recons = gdn_ae.SampleFromCode(tfsession, gdn_codes)

#... and that's about it for now there may be other useful functionality
# that I left out, but you can poke around in models.py and utils/ to see
# for yourself :)
