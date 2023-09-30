# Copyright 2020 Vladislav Lialin and Skillfactory LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Train a neural network classifier."""

import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import datasets
import toml
import wandb

import config  # python file with default hyperparameters

from tqdm.auto import tqdm

from nn_classifier import utils, data_utils
from nn_classifier.modelling import FcnBinaryClassifier

import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer    
  
# Optional
# wandb.watch(model)

wandb.init(project="hw3")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # fmt: off
    # preprocessing
    parser.add_argument("--max_vocab_size", default=31, type=int,
                        help="maximum size of the vocabulary")

    # model
    parser.add_argument("--hidden_size", default=32, type=int,
                        help="size of the intermediate layer in the network")
    # note that we can't use action='store_true' here or this won't work with wandb sweeps
    parser.add_argument("--use_batch_norm", default=False, type=lambda s: s.lower() == 'true')
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="L2 regularization parameter.")
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="Learning rate")

    # training
    parser.add_argument("--batch_size", default=64, type=int,
                        help="number of examples in a single batch")
    parser.add_argument("--max_epochs", default=5, type=int,
                        help="number of passes through the dataset during training")
    parser.add_argument("--early_stopping", default=1, type=int,
                        help="Stop training if the model does not improve the results after this many epochs")

    # misc
    parser.add_argument("--device", default=None, type=str,
                        help="device to train on, use GPU if available by default")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="a directory to save the model and config, do not save the model by default")
    parser.add_argument("--wandb_project", default="nlp_module_3_assignment",
                        help="wandb project name to log metrics to")
    # fmt: on

    args = parser.parse_args(args)

    return args


def main(args):
    """Train tokenizer, model and save them to a directory

    args should __only__ be used in this function or passed to a hyperparameter logger.
    Never propagate args further into your code - it causes complicated and tightly connected interfaces
    that are easy to modify, but impossible to read and use outside the main file.
    """

    if args.output_dir is not None and os.path.exists(args.output_dir):
        raise ValueError(f"output_dir {args.output_dir} already exists")

    # Initialize wandb as soon as possible to log all stdout to the cloud
    wandb.init(config=args)

    device = args.device
    # TASK 2.1: if device is not specified, set it to "cuda" if torch.cuda.is_available()
    # if cuda is not available, set device to "cpu"
    # Our implementation is 2 lines
    # YOUR CODE STARTS
    if torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')
    # YOUR CODE ENDS

    _device_description = "CPU" if device == "cpu" else "GPU"
    logger.info(f"Using {_device_description} for training")

    # Create dataset objects
    logger.info("Loading dataset")
    text_dataset = datasets.load_dataset("imdb")
    train_texts = text_dataset["train"]["text"]
    train_labels = text_dataset["train"]["label"]

    tokenizer = utils.make_whitespace_tokenizer(train_texts, max_vocab_size=args.max_vocab_size)
    train_dataset = data_utils.CountDataset(
        train_texts,
        tokenizer=tokenizer,
        labels=train_labels,
    )

    test_dataset = data_utils.CountDataset(
        text_dataset["test"]["text"], tokenizer, text_dataset["test"]["label"]
    )

    # It is very important to shuffle the training set
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model and optimizer
    input_size = tokenizer.get_vocab_size()
    model = FcnBinaryClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        dropout_prob=args.dropout,
        use_batch_norm=args.use_batch_norm,
    )
    model = model.to(device)
    wandb.watch(model)

    # TASK 2.2: Create AdamW optimizer (not Adam)
    # and provide learning rate and weight decay parameters to it
    # Our implementation is 1 line
    # YOUR CODE STARTS 
    optimizer = AdamW(learning_rate=1e-1, weight_decay=0.8)
    # YOUR CODE ENDS

    # Initialize current best accuracy as 0 for early stopping
    best_acc = 0
    epochs_without_improvement = 0  # training stops when this is larger than args.early_stopping

    # if args.output_dir is specified, create it and save args as a toml file
    # toml is a more flexible, readable and error-prone alternative to yaml and json

    if args.output_dir is not None:
        os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, "args.toml"), "w") as f:
            toml.dump(vars(args), f)
        tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))

    logger.info("Starting training")

    for _ in tqdm(range(args.max_epochs), desc="Epochs"):
        for x, y in dataloader:
            # TASK 2.3a: Define the training loop
            # 1. Move and and y to the device you are using for training
            # 2. Get class probabilites using model
            # 3. Calculate loss using F.binary_cross_entropy
            # 4. Zero out the cashed gradients from the previous iteration
            # 4. Backpropagate the loss
            # 5. Update the parameters
            # Our implementation is 7 lines
            
            # YOUR CODE STARTS
            x, y = x.to(device), y.to(device)
            probs = F.softmax(model(x), dim=1)            
            loss = F.binary_cross_entropy(torch.log(probs), y)
            optimizer = optim.SGD([x, y], lr=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # YOUR CODE ENDS

            wandb.log(
                {
                    "train_acc": utils.accuracy(probs, y),
                    "train_loss": loss,
                }
            )

        # Task 2.3b: Evaluate the model on the test set
        # Use utils.evaluate_model to get it and wandb.log to log it as "test_acc"
        # Our implementation is 2 lines
        # YOUR CODE STARTS
        test_acc = utils.evaluate_model(model, dataloader)
        wandb.log({"test_acc": test_acc})
        # YOUR CODE ENDS

        # TASK 2.4: if output_dir is provided and test accuracy is better than the current best accuracy
        # save the model to output_dir/model_checkpoint.pt
        # use os.path.join to write code transferable between Linux/Mac and Windows
        # extract save model.state_dict() using torch.save
        # set epochs_without_improvement to zero.
        # Remember to update best_acc even if output_dir is not provided.
        # Stop training (use break) if epochs_without_improvement > early_stopping
        # Before that use the logger.info to indicate that the training stopped early.
        # Our implementation is 12 lines
        # YOUR CODE STARTS
        if args.output_dir is not None and test_acc > best_acc:
            try:
                os.makedirs(os.path.join(args.output_dir, "model_checkpoint.pt"))
            except OSError:
                pass
            output = open(os.path.join(args.output_dir, "model_checkpoint.pt"), mode="wb")
            torch.save(model.state_dict(), output)
            output.close()
            epochs_without_improvement = 0
            early_stopping = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
            if epochs_without_improvement > early_stopping:
                break
        # YOUR CODE ENDS

    # Log the best accuracy as a summary so that wandb would use it instead of the final value
    wandb.run.summary["test_acc"] = best_acc

    logger.info("Training is finished!")


if __name__ == "__main__":
    args = parse_args()
    main(args)