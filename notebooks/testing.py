import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import argparse
from models.Framework import Framework
from utils.args import get_test_args
from data.dummy import create_dummy_data

args = get_test_args()
print(args)
model = Framework(args)
