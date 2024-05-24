{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import scipy.sparse as sp\n",
    "import os, sys, time\n",
    "from tqdm import tqdm\n",
    "sys.path.append(\"/home/hadh2/projects/cross_domain/cross_domain/PPGN\")\n",
    "from model import PPGN\n",
    "from multiprocessing import Pool\n",
    "import tensorflow as tf\n",
    "from utils import metrics\n",
    "from data.dataset import Dataset\n",
    "import argparse\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "usage: ipykernel_launcher.py [-h] [--gpu_device GPU_DEVICE]\n",
      "                             [--cross_data_rebuild CROSS_DATA_REBUILD]\n",
      "                             [--data_rebuild DATA_REBUILD]\n",
      "                             [--mat_rebuild MAT_REBUILD]\n",
      "                             [--processor_num PROCESSOR_NUM]\n",
      "                             [--batch_size BATCH_SIZE]\n",
      "                             [--train_neg_num TRAIN_NEG_NUM]\n",
      "                             [--test_size TEST_SIZE]\n",
      "                             [--test_neg_num TEST_NEG_NUM] [--epochs EPOCHS]\n",
      "                             [--gnn_layers GNN_LAYERS]\n",
      "                             [--mlp_layers MLP_LAYERS]\n",
      "                             [--embedding_size EMBEDDING_SIZE] [--topK TOPK]\n",
      "                             [--regularizer_rate REGULARIZER_RATE] [--lr LR]\n",
      "                             [--dropout_message DROPOUT_MESSAGE]\n",
      "                             [--NCForMF NCFORMF]\n",
      "ipykernel_launcher.py: error: unrecognized arguments: --f=/home/hadh2/.local/share/jupyter/runtime/kernel-v2-8900215pLnLbwDuS42.json\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "2",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[0;31mSystemExit\u001b[0m\u001b[0;31m:\u001b[0m 2\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/hadh2/anaconda3/envs/tf_env/lib/python3.9/site-packages/IPython/core/interactiveshell.py:3513: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "args_dict = {\n",
    "    'gpu_device': 6,\n",
    "    'cross_data_rebuild': False,\n",
    "    'data_rebuild': False,\n",
    "    'mat_rebuild': False,\n",
    "    'processor_num': 50,\n",
    "    'batch_size': 1024,\n",
    "    'train_neg_num': 0,\n",
    "    'test_size': 4,\n",
    "    'test_neg_num': 99,\n",
    "    'epochs': 50,\n",
    "    'gnn_layers': [128, 64, 32, 16, 8],\n",
    "    'mlp_layers': [32, 16, 8],\n",
    "    'embedding_size': 64,\n",
    "    'topK': 10,\n",
    "    'regularizer_rate': 0.01,\n",
    "    'lr': 0.001,\n",
    "    'dropout_message': 0.3,\n",
    "    'NCForMF': 'NCF'\n",
    "}\n",
    "\n",
    "parser = argparse.ArgumentParser()\n",
    "\n",
    "for arg_name, arg_value in args_dict.items():\n",
    "    parser.add_argument(f'--{arg_name}', type=type(arg_value), default=arg_value)\n",
    "\n",
    "args = parser.parse_args()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'args' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[4], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m dataset_s \u001b[38;5;241m=\u001b[39m Dataset(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m/home/hadh2/projects/cross_domain/cross_domain/PPGN/data/CDs_and_Vinyl.csv\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[43margs\u001b[49m)\n\u001b[1;32m      2\u001b[0m dataset_t \u001b[38;5;241m=\u001b[39m Dataset(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m/home/hadh2/projects/cross_domain/cross_domain/PPGN/data/Digital_Music.csv\u001b[39m\u001b[38;5;124m'\u001b[39m, args)\n",
      "\u001b[0;31mNameError\u001b[0m: name 'args' is not defined"
     ]
    }
   ],
   "source": [
    "dataset_s = Dataset('/home/hadh2/projects/cross_domain/cross_domain/PPGN/data/CDs_and_Vinyl.csv', args)\n",
    "dataset_t = Dataset('/home/hadh2/projects/cross_domain/cross_domain/PPGN/data/Digital_Music.csv', args)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_dict = np.load(train_path, allow_pickle=True).item()\n",
    "test_dict = np.load(test_path, allow_pickle=True).item()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_path = dataset_s.path + '/cross_' + '_'.join([dataset_s.name, dataset_t.name]) + '_train.npy'\n",
    "test_path = dataset_t.path + '/cross_' + '_'.join([dataset_s.name, dataset_t.name]) + '_test.npy'"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "tf_env",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
