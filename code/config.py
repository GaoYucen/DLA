# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description="config of models")

# look
parser.add_argument('--look_back', type=int, default=24, help='look_back')
parser.add_argument('--look_forward', type=int, default=6, help='look_forward')

# long look
parser.add_argument('--long_look_back', type=int, default=60, help='long look_back')
parser.add_argument('--long_look_forward', type=int, default=30, help='long look_forward')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed