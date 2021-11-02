import argparse
import random
import numpy as np
import torch
import os


def get_argparser():
    """
    Sets up and returns an argparser for the training procedure
    """
    parser = argparse.ArgumentParser()

    # Data related args
    parser.add_argument('--data_folder', type=str, default="data", help="The folder in which the data is located")
    parser.add_argument('--data_filename', type=str, default="", help="The name of the data file, minus the extension")

    # General simulation settings
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # Training settings
    parser.add_argument("--train_steps", default=1, type=int, help="Number of training steps to perform")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    return parser


def set_seed(seed=42):
    """
    Sets all relevant seeds to a fixed value
    """
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
