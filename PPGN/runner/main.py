import numpy as np

import argparse
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.dataset import Dataset
from train import train
import tensorflow as tf

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_device', type=int, default=6,
                    help='choose which gpu to run')
parser.add_argument('--cross_data_rebuild', type=bool, default=False,
                    help='whether to rebuild cross data')
parser.add_argument('--data_rebuild', type=bool, default=False,
                    help='whether to rebuild train/test dataset')
parser.add_argument('--mat_rebuild', type=bool, default=False,
                    help='whether to rebuild` adjacent mat')
parser.add_argument('--processor_num', type=int, default=50,
                    help='number of processors when preprocessing data')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='size of mini-batch')
# parser.add_argument('--train_neg_num', type=int, default=0,
#                     help='number of negative samples per training positive sample')
parser.add_argument('--test_size', type=int, default=4,
                    help='size of sampled test data')
parser.add_argument('--test_neg_num', type=int, default=99,
                    help='number of negative samples for test')
parser.add_argument('--epochs', type=int, default=50,
                    help='the number of epochs')
parser.add_argument('--gnn_layers', nargs='?', default=[128,64,32,16,8],
                    help='the unit list of layers')
parser.add_argument('--mlp_layers', nargs='?', default=[32,16,8],
                    help='the unit list of layers')
parser.add_argument('--embedding_size', type=int, default=64,
                    help='the size for embedding user and item')
parser.add_argument('--topK', type=int, default=10,
                    help='topk for evaluation')
parser.add_argument('--regularizer_rate', type=float, default=0.01,
                    help='the regularizer rate')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--dropout_message', type=float, default=0.3,
                    help='dropout rate of message')
parser.add_argument('--NCForMF', type=str, default='NCF',
                    help='method to propagate embeddings')

args = parser.parse_args()


if __name__ == '__main__':
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    dataset_s = Dataset('/home/hadh2/projects/cross_domain/PPGN/data/CDs_and_Vinyl.csv', args)
    dataset_t = Dataset('/home/hadh2/projects/cross_domain/PPGN/data/Digital_Music.csv', args)
    train(dataset_s, dataset_t, args)
