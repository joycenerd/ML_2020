import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cuda_devices', type=int, default=0, help='gpu device number')
    parser.add_argument('--epoch', type=int, default=100, help='number of epoch to train')
    parser.add_argument('--lr',default=0.01,help='initial learning rate')
    parser.add_argument('--mode',default='leave-one-out',choices=['leave-one-out', 'five-fold'], help='leave-one-out or five-fold')
    return parser.parse_args()
