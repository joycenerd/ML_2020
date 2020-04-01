import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cuda_devices', type=int, default=0, help='gpu device number')
    return parser.parse_args()
