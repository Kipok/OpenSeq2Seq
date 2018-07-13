# pylint: skip-file
from open_seq2seq.models import Image2Label
from open_seq2seq.encoders.cnn_encoder import CNNEncoder
from open_seq2seq.decoders import FullyConnectedDecoder
from open_seq2seq.losses import CrossEntropyLoss
from open_seq2seq.data import ImagenetDataLayer
from open_seq2seq.optimizers.lr_policies import piecewise_constant
import tensorflow as tf


base_model = Image2Label

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_epochs": 100,

  "num_gpus": 8,
  "batch_size_per_gpu": 32,
  "dtype": tf.float32,

  "save_summaries_steps": 2000,
  "print_loss_steps": 100,
  "print_samples_steps": 2000,
  "eval_steps": 5000,
  "save_checkpoint_steps": 5000,
  "logdir": "experiments/resnet50-imagenet",

  "optimizer": "Momentum",
  "optimizer_params": {
    "momentum": 0.90,
  },
  "lr_policy": piecewise_constant,
  "lr_policy_params": {
    "learning_rate": 0.1,
    "boundaries": [30, 60, 80, 90],
    "decay_rates": [0.1, 0.01, 0.001, 1e-4],
  },

  "initializer": tf.variance_scaling_initializer,

  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 0.0001,
  },
  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],
  "encoder": CNNEncoder,
  "encoder_params": {
    'data_format': 'channels_first',
    'cnn_layers': [
      (conv2d_fixed_padding, {'filters': 64, 'kernel_size': 7, 'conv_stride': 2}),
      (tf.layers.max_pooling2d, {"pool_size": 3, "strides": 2, "padding": "SAME"}),

      (tf.layers.batch_normalization, {'momentum': 0.997, "epsilon": 1e-5}),
      (tf.nn.relu, {}),

    ],
  },
  "decoder": FullyConnectedDecoder,
  "decoder_params": {
    "output_dim": 1000,
  },
  "loss": CrossEntropyLoss,
  "data_layer": ImagenetDataLayer,
  "data_layer_params": {
    "data_dir": "data/tf-imagenet",
    "image_size": 224,
  },
}
