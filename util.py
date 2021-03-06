import argparse
import random
import numpy as np
import torch
import os
from provided.generator_utils import decode


def get_argparser():
    """
    Sets up and returns an argparser for the training procedure
    """
    parser = argparse.ArgumentParser()

    # Data related args
    parser.add_argument("--data_folder", type=str, default="data", help="The folder in which the data is located")
    parser.add_argument("--data_filename", type=str, default="", help="The name of the data file, minus the extension")

    parser.add_argument("--output_folder", type=str, default="out", help="The folder in which all output is stored")

    parser.add_argument("--max_source_length", type=int, default=64, help="The maximum length in tokens of the source")
    parser.add_argument("--max_target_length", type=int, default=64, help="The maximum length in tokens of the target")

    parser.add_argument("--model_params", type=str, default="", help="The location of the model parameters (bin file)")

    # General simulation settings
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--device", type=str, default="cpu", help="random seed for initialization")
    parser.add_argument("--mode", type=str, default="train", help="Whether to run in train mode or eval mode")
    parser.add_argument("--beam", type=int, default=10, help="The size of the beam search")

    # Training settings
    parser.add_argument("--train_steps", default=1, type=int, help="Number of training steps to perform")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument(
        "--dev_filename",
        default=None,
        type=str,
        help="The dev filename. Should contain the .jsonl files for this task.",
    )

    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument(
        "--test_filename",
        default=None,
        type=str,
        help="The test filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--source",
        default="en",
        type=str,
        help="The source language (for file extension)",
    )
    parser.add_argument(
        "--target",
        default="sparql",
        type=str,
        help="The target language (for file extension)",
    )

    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )

    parser.add_argument(
        "--delta_es",
        type=float,
        default=0.005,
        help="The delta parameter for ES",
    )

    parser.add_argument(
        "--do_early_stopping", action="store_true", help="Whether to do ES on dev set."
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs for patience in ES",
    )

    # Args for evaluator
    parser.add_argument("--gold", dest="gold", required=True,
                        help="Golden standard file for evaluation")
    parser.add_argument("--output", dest="output", required=True,
                        help="Output file of our system, for evaluation")

    return parser


def set_seed(seed=42):
    """
    Sets all relevant seeds to a fixed value
    """
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(model, folder):
    output_dir = folder

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)


def revert_query(sparql):
    return decode(sparql)

def space_mapper(queries):
    res = []
    for i in range(len(queries)):
        queries[i] = queries[i].replace("< ", "<").replace(" >", ">").replace(" _ ", "_").replace(" : ", ":")

    return queries
