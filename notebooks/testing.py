import os, sys
sys.path.append(os.path.abspath('..'))

import torch
import argparse
from models.Framework import Framework
from utils.args import get_test_args
from data.dummy import get_dataloader

args = get_test_args()
print(args)
model = Framework(args)
train_loader, val_loader, test_locdader = get_dataloader(args)
