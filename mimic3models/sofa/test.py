import numpy as np
import argparse
import os

from mimic3benchmark.readers import SepsisSOFAReader
from mimic3models.preprocessing import Discretizer
from mimic3models import common_utils

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--deep_supervision', dest='deep_supervision', action='store_true')
parser.set_defaults(deep_supervision=False)
parser.add_argument('--data', type=str, help='Path to the data of SOFA')
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
args = parser.parse_args()
print(args)

# Build readers
train_reader = SepsisSOFAReader(dataset_dir=os.path.join(args.data, 'train'),
                                  listfile=os.path.join(args.data, 'train_listfile.csv'))
val_reader = SepsisSOFAReader(dataset_dir=os.path.join(args.data, 'train'),
                                listfile=os.path.join(args.data, 'val_listfile.csv'))

discretizer = Discretizer(timestep=args.timestep,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

# Get first example
first_example = train_reader.read_example(0)

# Debug: Check alignment
print("Channels in your data:", first_example["header"])
print("\nChannels in discretizer config:", discretizer._id_to_channel)
print("\nChannels in data but NOT in config:", 
      set(first_example["header"][1:]) - set(discretizer._id_to_channel))
print("\nChannels in config but NOT in data:", 
      set(discretizer._id_to_channel) - set(first_example["header"][1:]))

# Transform with the correct header
discretizer_header = discretizer.transform(
    first_example["X"],
    header=first_example["header"]  # ← THIS IS THE FIX
)[1].split(',')

cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

print("\nTransformed successfully!")
print(f"Number of features after discretization: {len(discretizer_header)}")