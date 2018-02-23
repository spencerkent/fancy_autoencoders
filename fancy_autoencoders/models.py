"""
Some simple wrappers on TensorFlow graphs that do various kinds of autoencoding
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from utils.plotting import plot_weights
from utils.plotting import analyze_code
from utils.plotting import plot_reconstructions

class Autoencoder(object):
  """
  This is the base class of a simple autoencoder
  """
  ############################################################################
  # some private functions we'll use in derived classes to construct the graph
  def _add_inputs(self, image_params, hiddenL_params):
    self._image_height = image_params['height']
    self._image_width = image_params['width']
    self._image_depth = image_params['depth']
    self._image_flatsize = (self._image_height * self._image_width *
                            self._image_depth)
    self._c_size = hiddenL_params['code_size']

    self.images = tf.placeholder(tf.float32,
                                 shape=[None, self._image_flatsize],
                                 name='InputPatches')
  def _add_common_encoding(self, hiddenL_params):
    """
    This is shared among all fancy autoencoder models. Just a linear transform
    """
    self.weight_hidden = tf.Variable(
        tf.truncated_normal([self._image_flatsize,
                             self._c_size],
                            stddev=hiddenL_params['weight_init_std']),
        name='WeightInputToHidden', dtype=tf.float32)

    # Is it a a fair comparison if we get a bias to train too?
    self.bias_hidden = tf.Variable(tf.constant(0.0, shape=[self._c_size]),
                                   dtype=tf.float32)
    self.neuron_currents = tf.matmul(self.images, self.weight_hidden) + \
                           self.bias_hidden
    # self.neuron_currents = tf.matmul(self.images, self.weight_hidden)

  def _add_common_decoding(self, outputL_params):
    """
    This is shared among all fancy autoencoder models. Just a linear transform
    """
    self.weight_output = tf.Variable(
        tf.truncated_normal([self._c_size,
                             self._image_flatsize],
                            stddev=outputL_params['weight_init_std']),
        name='WeightHiddenToOutput', dtype=tf.float32)

    # Is it a a fair comparison if we get a bias to train too?
    self.bias_output = tf.Variable(
      tf.constant(0.0, shape=[self._image_flatsize]), dtype=tf.float32)
    self.output = tf.matmul(self.neuron_activations_inv, self.weight_output) + \
                   self.bias_output
    # self.output = tf.matmul(self.neuron_activations_inv, self.weight_output)

  def _add_loss(self, outputL_params, default_opt_step):
    """
    Shared among all fancy autoencoder models
    """
    self._loss_type = outputL_params['loss_type']

    if self._loss_type == 'l2':
      self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
        tf.square(tf.subtract(self.output, self.images)), axis=1)))

    elif self._loss_type == 'l2_recon_l1_sparsity':
      self.sparsity_wt = tf.constant(outputL_params['init_sparsity_weight'])
      self.loss = (
          tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(
            tf.subtract(self.output, self.images)), axis=1))) +
          self.sparsity_wt * tf.reduce_mean(
            tf.reduce_sum(tf.abs(self.neuron_activations), axis=1)))

    elif self._loss_type == 'l2recon_l1sparsity_wd':
      self.sparsity_wt = tf.constant(outputL_params['init_sparsity_weight'])
      self.weight_decay_wt = tf.constant(outputL_params['init_wd_weight'])
      self.loss = (
          tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(
            tf.subtract(self.output, self.images)), axis=1))) +
          self.sparsity_wt * tf.reduce_mean(
            tf.reduce_sum(tf.abs(self.neuron_activations), axis=1)) +
          self.weight_decay_wt * tf.sqrt(tf.reduce_sum(
            tf.square(self.weight_hidden))) +
          self.weight_decay_wt * tf.sqrt(tf.reduce_sum(
            tf.square(self.weight_output))))

    self.opt_step = tf.constant(default_opt_step)
    self.optimizer_step = \
        tf.train.AdamOptimizer(self.opt_step).minimize(self.loss)

  def _add_remaining_misc(self):
    self.variables_saver = None
    # used for loading in a pretrained model
    self.learned_params = []
    for section in ['Encoding', 'Decoding']:
      self.learned_params = self.learned_params + \
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                          scope=section)


  def StartSession(self, checkpoint_file=None):
    """
    Starts a TensorFlow session and optionally loads in parameters of the model

    Parameters
    ----------
    checkpoint_file : str, optional
        If given we load the session from a checkpoint file, it None, then start
        a fresh session

    Returns
    -------
    tfsess : tensorflow.Session
        A session object so that we can do stuff with the model, like train it.
    """
    print("Okay, let's fire things up")
    tfsess = tf.Session()
    tfsess.run(tf.global_variables_initializer())
    if checkpoint_file is not None:
      if self.variables_saver is None:
        self.variables_saver = \
          tf.train.Saver(var_list=self.learned_params)
      self.variables_saver.restore(tfsess, checkpoint_file)

    return tfsess


  def SaveSession(self, sess, save_dir, checkpoint_name):
    """
    Saves a checkpoint file that preserves the full state of the session

    Parameters
    ----------
    sess : tensorflow.Session
        The current tensorflow session whose state to save
    save_dir : str
        The relative path of the save directory
    """
    print("Saving this session...")
    if not os.path.isdir(os.path.abspath(save_dir)):
      os.mkdir(save_dir)
    if self.variables_saver is None:
      self.variables_saver = \
          tf.train.Saver(var_list=self.learned_params)
    self.variables_saver.save(sess, save_dir + '/' + checkpoint_name + '.ckpt')


  def Train(self, sess, train_images, val_images, batch_size,
            step_schedule, snapshot_progress=False, logging_dir=None,
            report_every=100, max_epochs=100):
    """
    Trains the autoencoder

    Parameters
    ----------
    sess : tensflow.Session
        The (already initialized) TF session to do training on. Could be
        pretrained already
    train_images : ndarray
        Each row indexes a image of size self._image_flatsize. Images cast to
        float32
    val_images : ndarray
        Each row indexes a image of size self._image_flatsize. Images cast to
        float32. We'll check these to determine when to stop training
    step_schedule : dictionary
        Contains iterations numbers at which to set (reset) the ADAM step size
        Example: {0: 0.001, 1e4: 0.0001, 1e8: 0.00001}
    shapshot_progress : bool, optional
        If true, we periodically save some plots to disk so we can check
        training progress. These will overwrite eachother. We plot
        the current encoding and decoding weights for if we're
        shooting for oriented filters or something like that. Also plot the
        histogram of hidden unit activations, and some sample reconstructions.
        Will save them in directory specified by logging_dir
    the rest are self-explanatory...

    Returns
    -------
    training_report : dict
        'terminating_iteration' : int
        'training_loss' : dict
          'iternum' : int
          'value' : list
        'validation_loss' : dict
          'iternum' : int
          'value' : list
    """
    assert train_images.shape[1] == self._image_flatsize
    assert val_images.shape[1] == self._image_flatsize
    assert 0 in step_schedule, 'please indicate initial ADAM step size'
    ts_size = train_images.shape[0]
    vs_size = val_images.shape[0]


    self.training_loss = []
    self.training_loss_report_step = report_every
    self.validation_metric = []
    self.validation_metric_report_step = report_every * 10
    self.snapshot_report_step = report_every * 100
    val_metric_improved = True
    burn_in_period = max_epochs // 10  # don't check the validation error
                                      # until the net has been training for
                                      # this long

    batch_idx_perm = np.arange(train_images.shape[0])
    b_idx = 0
    epochs_completed = 0
    iternum = 0
    while epochs_completed != max_epochs and val_metric_improved:

      if b_idx + batch_size > ts_size:
        # we've come to the end of an epoch
        np.random.shuffle(batch_idx_perm)
        b_idx = 0
        epochs_completed += 1

      batch_images = train_images[batch_idx_perm[b_idx:b_idx+batch_size]]
      b_idx += batch_size

      if iternum in step_schedule:
        current_step_size = step_schedule[iternum]

      self.optimizer_step.run(session=sess,
                              feed_dict={self.images: batch_images,
                                         self.opt_step: current_step_size})

      if iternum % report_every == 0:
        cost = sess.run(self.loss, feed_dict={self.images: batch_images})
        self.training_loss.append(cost)

        print("------------------------------")
        print("Iternum ", iternum, " of epoch ", epochs_completed)
        print(self._loss_type + " loss is ", cost, " on the training data")

      # Check the validation error
      if iternum % (self.validation_metric_report_step) == 0:
        val_mean  = self.Test(sess, val_images)

        if epochs_completed >= burn_in_period:
          moving_avg = np.mean(self.validation_metric[-3:])
          val_metric_improved = val_mean < moving_avg

        self.validation_metric.append(val_mean)

      # Possibly save some progress plots to disk
      if snapshot_progress and iternum % (self.snapshot_report_step) == 0:

        enc_w, dec_w = self.GetEncDecWeights(sess)
        enc_w_figs = plot_weights(enc_w,
            {'height': self._image_height, 'width': self._image_width},
            enc_dec='encoding', single_img=True)
        dec_w_figs = plot_weights(
            dec_w, {'height': self._image_height, 'width': self._image_width},
            enc_dec='decoding',
            single_img=True)

        for fig_idx in range(len(enc_w_figs)):
          enc_w_figs[fig_idx].savefig(
              logging_dir + 'snapshot_enc_weights_' + str(fig_idx) + '.png')
          plt.close(enc_w_figs[fig_idx])
        for fig_idx in range(len(dec_w_figs)):
          dec_w_figs[fig_idx].savefig(
              logging_dir + 'snapshot_dec_weights_' + str(fig_idx) + '.png')
          plt.close(dec_w_figs[fig_idx])
        rand_img_inds = np.random.choice(np.arange(val_images.shape[0]), 9,
                                         replace=False)
        activation_plots = analyze_code(self.GetCodes(sess, val_images),
                                        rand_img_inds)
        activation_plots[0].savefig(logging_dir + 'snapshot_act_hist.png')
        plt.close(activation_plots[0])
        plt.close(activation_plots[1])

        snap_loss, snap_recons = self.Test(sess, val_images, True)
        reconplots = plot_reconstructions(val_images, snap_recons,
            {'height': self._image_height, 'width': self._image_width},
            rand_img_inds)
        reconplots[0].savefig(logging_dir + 'snapshot_gt_imgs.png')
        reconplots[1].savefig(logging_dir + 'snapshot_recon_imgs.png')
        plt.close(reconplots[0])
        plt.close(reconplots[1])

      iternum += 1

    print("Finished!")
    if epochs_completed != max_epochs:
      print("Terminated early for some reason")
      print("Termination was at iteration number {}, (epoch number {})"
            "".format(iternum, epochs_completed))

    # just grab the iteration numbers that correspond to each console log entry
    i_step = self.training_loss_report_step
    xt = np.arange(0, len(self.training_loss)*i_step, i_step)
    i_step = self.validation_metric_report_step
    xv = np.arange(0, len(self.validation_metric)*i_step, i_step)

    training_report = {'terminating_iteration': iternum - 1,
                       'training_loss':
                           {'iternum': xt,
                            'value': self.training_loss},
                       'validation_metric':
                           {'iternum': xv,
                            'value': self.validation_metric}}
    return training_report


  def Test(self, sess, test_or_val_images, return_reconstructions=False):
    """
    Pass some test images through the model

    Will return the mean and std of our requested loss function between the
    ground truth images and the output of the network

    Parameters
    ----------
    sess : tf.Session object
        The graph session to run the dataset through
    test_or_val_images : ndarray
    return_reconstructions : bool, optional
        If True return the reconstructions. Default False

    Returns
    -------
    loss : float
        The mean across the given dataset of our chosen loss function
    reconstructions : ndarray, (if return_reconstructions==True)
        Same size as test_or_val_images
    """
    assert test_or_val_images.shape[1] == self._image_flatsize
    print("*****************************************************")
    print("Running {} test images through the autoencoder"
          "".format(test_or_val_images.shape[0]))
    loss, reconstructions = sess.run(
        [self.loss, self.output], feed_dict={self.images: test_or_val_images})
    print("The mean of the test loss was ", loss)

    # we may want to know how good the reconstructions are irrespective of the
    # loss so we'll compute the MSE manually
    mse = np.mean(np.square(reconstructions - test_or_val_images), axis=1)
    psnrs = 10 * np.log10(1. / mse)
    print("The mean PSNR of the reconstructions is ", np.mean(psnrs), ' dB')
    print("*****************************************************")


    if return_reconstructions:
      return loss, reconstructions
    else:
      return loss


  def GetEncDecWeights(self, sess):
    return (self.weight_hidden.eval(session=sess),
            self.weight_output.eval(session=sess))


  def GetCodes(self, sess, inputs):
    return sess.run(self.neuron_activations, feed_dict={self.images: inputs})



####################################
# now we define some specific models
####################################



class GDN_autoencoder(Autoencoder):
  """
  A Generalized divisive normalization-based autoencoder

  Parameters
  ----------
  hiddenL_params : dictionary
      'weight_init_std' : float
        The standard deviation of the random gaussian initialization for
        feedforward weights
      'code_size' : int
        Number of neurons in the hidden layer AKA code length
  outputL_params : dictionary
      'weight_init_std' : float
        The standard deviation of the random gaussian initialization for
        feedforward weights
      'loss_type' : str
        Currently one of {'l2', 'l2recon_l1sparsity','l2recon_l1sparsity_wd'}
      'init_sparsity_weight' : float, optional
        The value of \lambda, the weight on the second term when we are
        using the l1-regularized sparse loss
      'init_wd_weight' : float, optional
        The value of \beta, the weight on the third term when we are
        using the l1-regularized sparse loss WITH WEIGHT DECAY
  image_params : dictionary
      Keys are 'height', 'width', 'depth'
  default_opt_step : float
      The default value to instantiate ADAM with. We can change this at training
      time by passing the step_schedule parameter to this object's Train method
  """
  def __init__(self, hiddenL_params, outputL_params,
               image_params, default_opt_step):

    with tf.name_scope('Input'):
      self._add_inputs(image_params, hiddenL_params)

    with tf.name_scope('Encoding'):
      self._add_common_encoding(hiddenL_params)

      ## we apply GDN nonlinearity to self.neuron_currents
      self.gdn_weights = tf.Variable(
          tf.multiply(tf.ones([self._c_size, self._c_size], tf.float32), 1e-3),
          name='GDN_weights')

      self.gdn_biases = tf.Variable(
          tf.multiply(tf.ones([self._c_size], tf.float32), 1e-3),
          name='GDN_biases')

      self.neuron_activations = self._babysat_GDN(
        self.neuron_currents, self.gdn_weights, self.gdn_biases,
        'forward', 'GDN_codes')

    with tf.name_scope('Decoding'):
      ## Invert the GDN nonlinearity
      self.inv_gdn_weights = tf.Variable(
          tf.multiply(tf.ones([self._c_size, self._c_size], tf.float32), 1e-3),
          name='Inv_GDN_weights')

      self.inv_gdn_biases = tf.Variable(
          tf.multiply(tf.ones([self._c_size], tf.float32), 1e-3),
          name='Inv_GDN_biases')

      self.neuron_activations_inv = self._babysat_GDN(
        self.neuron_activations, self.inv_gdn_weights, self.inv_gdn_biases,
        'inverse', 'Inv_GDN_currents')

      self._add_common_decoding(outputL_params)

    with tf.name_scope('Loss'):
      self._add_loss(outputL_params, default_opt_step)

    self._add_remaining_misc()


  @staticmethod
  def _vanilla_GDN(input_currents, gdn_weights, gdn_biases, direction,
                   gdn_layer_name):
    """
    Implements the generic GDN given in eq (3) & (4) of Balle's 2017 ICLR paper

    Parameters
    ----------
    input_currents : tf.Tensor
        The input currents to each neuron for a batch. B x H where B is the
        batch size and H is the number of neurons in the latent coding layer
        These are the 'w' variables in Balle's paper. Often denote by 'c' or
        'x' in the literature
    gdn_weights : tf.Tensor (Variable)
        An H x H weight matrix where H is the latent coding layer size.
        Denoted by \gamma in Balle's 2017 paper and
        often also 'h' or 'w' in the literature. Horizontal connections.
    gdn_biases : tf.Tensor (Variable)
        An H-dimensional bias vector, which is in the denominator of the
        gdn equation. Denoted by \beta in Balle's 2017 paper
    direction : str
        Either 'forward' or 'inverse'. Forward is the normal GDN. Inverse
        is the opposite of GDN but doesn't necessarily have to use the same
        weight an bias parameters...
    gdn_layer_name : str
        The TF name of the tensor that is returned

    Returns
    -------
    gdn_neuron_outputs : tf.Tensor
        A divisively normalized neuron output tensor the same shape as
        input_currents
    """
    if direction == 'forward':
      return tf.divide(input_currents,
                       tf.sqrt(gdn_biases + tf.matmul(tf.square(input_currents),
                                                      gdn_weights)),
                       name=gdn_layer_name)
    elif direction == 'inverse':
      return tf.multiply(input_currents,
          tf.sqrt(gdn_biases + tf.matmul(tf.square(input_currents),
                                         gdn_weights)), name=gdn_layer_name)
    else:
      raise KeyError('Unrecognized GDN direction ' + direction)


  @staticmethod
  def _babysat_GDN(input_currents, gdn_weights, gdn_biases, direction,
                   gdn_layer_name):
    """
    Implements a special 'babysat' version of GDN

    Attempts to avoid instability via a number of hacks that aren't well
    justfied but seem to make training more reasonable. Based on CAE by
    Dylan and Ryan

    Parameters
    ----------
    input_currents : tf.Tensor
        The input currents to each neuron for a batch. B x H where B is the
        batch size and H is the number of neurons in the latent coding layer
        These are the 'w' variables in Balle's paper. Often denote by 'c' or
        'x' in the literature
    gdn_weights : tf.Tensor (Variable)
        An H x H weight matrix where H is the latent coding layer size.
        Denoted by \gamma in Balle's 2017 paper and
        often also 'h' or 'w' in the literature. Horizontal connections.
    gdn_biases : tf.Tensor (Variable)
        An H-dimensional bias vector, which is in the denominator of the
        gdn equation. Denoted by \beta in Balle's 2017 paper
    direction : str
        Either 'forward' or 'inverse'. Forward is the normal GDN. Inverse
        is the opposite of GDN but doesn't necessarily have to use the same
        weight an bias parameters...
    gdn_layer_name : str
        The TF name of the tensor that is returned

    Returns
    -------
    gdn_neuron_outputs : tf.Tensor
        A divisively normalized neuron output tensor the same shape as
        input_currents
    """
    # hack #1: clip weight values that are less than 0.001
    clipped_gdn_weights = tf.where(tf.less(gdn_weights, tf.constant(1e-3)),
                                   1e-3 * tf.ones_like(gdn_weights),
                                   gdn_weights)
    # hack #2: clip bias values that are less than 0.001
    clipped_gdn_biases = tf.where(tf.less(gdn_biases, tf.constant(1e-3)),
                                  1e-3 * tf.ones_like(gdn_biases), gdn_biases)
    # hack #3: make the actual weight matrix gdn_weights + gdn_weights.T
    symmetric_weights = tf.add(clipped_gdn_weights,
                               tf.transpose(clipped_gdn_weights))

    gdn_rescaling = tf.sqrt(clipped_gdn_biases +
                            tf.matmul(tf.square(input_currents),
                                      symmetric_weights))
    if direction == 'forward':
      # hack #4: only apply GDN if denominator is sufficiently big
      return tf.where(tf.less(gdn_rescaling, tf.constant(1e-7)),
                      input_currents, tf.divide(input_currents, gdn_rescaling),
                      name=gdn_layer_name)
    elif direction == 'inverse':
      return tf.multiply(input_currents, gdn_rescaling, name=gdn_layer_name)
    else:
      raise KeyError('Unrecognized GDN direction ' + direction)


class RELU_autoencoder(Autoencoder):
  """
  A RELU-based autoencoder

  Parameters
  ----------
  hiddenL_params : dictionary
      'weight_init_std' : float
        The standard deviation of the random gaussian initialization for
        feedforward weights
      'code_size' : int
        Number of neurons in the hidden layer AKA code length
  outputL_params : dictionary
      'weight_init_std' : float
        The standard deviation of the random gaussian initialization for
        feedforward weights
      'loss_type' : str
        Currently one of {'l2', 'l2recon_l1sparsity','l2recon_l1sparsity_wd'}
      'init_sparsity_weight' : float, optional
        The value of \lambda, the weight on the second term when we are
        using the l1-regularized sparse loss
      'init_wd_weight' : float, optional
        The value of \beta, the weight on the third term when we are
        using the l1-regularized sparse loss WITH WEIGHT DECAY
  image_params : dictionary
      Keys are 'height', 'width', 'depth'
  default_opt_step : float
      The default value to instantiate ADAM with. We can change this at training
      time by passing the step_schedule parameter to this object's Train method
  """
  def __init__(self, hiddenL_params, outputL_params,
               image_params, default_opt_step):

    with tf.name_scope('Input'):
      self._add_inputs(image_params, hiddenL_params)

    with tf.name_scope('Encoding'):
      self._add_common_encoding(hiddenL_params)

      self.neuron_activations = tf.nn.relu(self.neuron_currents,
                                           name='relu_codes')
    with tf.name_scope('Decoding'):
      self.neuron_activations_inv = self.neuron_activations
      # no inversion of the nonlinearity in the RELU model

      self._add_common_decoding(outputL_params)

    with tf.name_scope('Loss'):
      self._add_loss(outputL_params, default_opt_step)

    self._add_remaining_misc()
