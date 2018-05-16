from open_seq2seq.models.image2label import ResNet
from open_seq2seq.data.image2label import ImagenetDataLayer

import sys
import os
sys.path.insert(0, os.path.abspath("tensorflow-models"))
from open_seq2seq.optimizers.lr_policies import poly_decay


batch_size_per_gpu = 32
num_gpus = 8

initial_lr = 1.0
larc_nu = 0.002
power = 1.0


base_model = ResNet

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_epochs": 25,

  "num_gpus": num_gpus,
  "batch_size_per_gpu": batch_size_per_gpu,

  "save_summaries_steps": 2000,
  "print_loss_steps": 2000,
  "print_samples_steps": 2000,
  "eval_steps": 5000,
  "save_checkpoint_steps": 5000,
  "logdir": "experiments/resnet50-imagenet",

  "optimizer": "Momentum",
  "optimizer_params": {
    "momentum": 0.90,
  },
  "larc_params": {
    "larc_nu": larc_nu,
  },
  "lr_policy": poly_decay,
  "lr_policy_params": {
    "power": power,
  },
  "learning_rate": initial_lr,

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],
}

train_params = {
  "data_layer": ImagenetDataLayer,
  "data_layer_params": {
    "data_dir": "data/tf-imagenet",
  },
}

eval_params = {
  "data_layer": ImagenetDataLayer,
  "data_layer_params": {
    "data_dir": "data/tf-imagenet",
  },
}

