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
test_patches_file = 'probably /media/tbell/sanborn/rd_analysis/inputs/field_100k_patches.p'
run_identifier = 'sparse_GDN'

model_params = pickle.load(open(logfile_directory + run_identifier +
                                '_model_params.p', 'rb'))


image_params = model_params['image_params']
hidden_layer_params = model_params['hidden_layer_params']
output_layer_params = model_params['output_layer_params']

# no more patch wrangling, the data we use is already AR-whitened and centered
test_dset = pickle.load(open(test_patches_file, 'rb'))


# okay let's load in the model
gdn_ae = GDN_autoencoder(hidden_layer_params, output_layer_params,
                         image_params)
tfsession = gdn_ae.StartSession(logfile_directory + run_identifier + '.ckpt')

# we can generate plots of the encoding and decoding weights or other things,
# see functionality in utils/plotting
encoding_weights, decoding_weights = gdn_ae.GetEncDecWeights(tfsession)
encoding_weight_figs = utils.plotting.plot_weights(
    encoding_weights, image_params, 'encoding', single_img=True, renormalize=True)
decoding_weight_figs = utils.plotting.plot_weights(
    decoding_weights, image_params, 'decoding', single_img=True, renormalize=True)


# we can run some inputs through the net with the Test method
_, gdn_reconstructions = gdn_ae.Test(tfsession, test_dset,
                                     return_reconstructions=True)

# we can fetch the 'GDN code for a set of inputs
gdn_codes = gdn_ae.GetCodes(tfsession, test_dset)


# display a random sample of the codes and corresponding reconstructions
some_rand_inds = np.random.choice(np.arange(test_dset.shape[0]), 16,
                                  replace=False)
codes_figs = utils.plotting.analyze_code(gdn_codes, some_rand_inds)
recon_figs = utils.plotting.plot_reconstructions(
    test_dset, gdn_reconstructions, image_params, some_rand_inds)


# we might want to fiddle with the codes a bit (quantize, etc) and then we
# can generate reconstructions from these new codes
different_recons = gdn_ae.SampleFromCode(tfsession, gdn_codes)

plt.show()
#... and that's about it for now there may be other useful functionality
# that I left out, but you can poke around in models.py and utils/ to see
# for yourself :)
