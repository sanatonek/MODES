
import sys
import os
import numpy as np
import argparse
sys.path.append('/home/sana/ml4h')
sys.path.append('/home/sana')
import tensorflow as tf
import json
from tensorflow import keras 

import matplotlib.pyplot as plt
import random
import seaborn as sns
sns.set_theme

from multimodal.fusion import DeMuReFusion
from multimodal.utils import *

tf.keras.callbacks.TerminateOnNaN()
tf.config.list_physical_devices('GPU')

parser = argparse.ArgumentParser(description='DeMuRe Fusion')
parser.add_argument('--log_path', default='/home/sana/multimodal/logs/out.txt')
parser.add_argument('--plot_path', default='/home/sana/multimodal/plots/')
parser.add_argument('--ckpt_path', default='/home/sana/multimodal/ckpts/')
parser.add_argument('--zm_size', default=400, type=int)
parser.add_argument('--zs_size', default=600, type=int)
parser.add_argument('--beta', default=1e-6)
parser.add_argument('--gamma', default=1)
parser.add_argument('--n_samples', default=1000, type=int)
parser.add_argument('--n_epochs', default=10, type=int)
parser.add_argument('--modality_names', nargs='+', default=['input_ecg_rest_median_raw_10_continuous', 'input_lax_4ch_heart_center_continuous'])


def main():
    args = parser.parse_args()
    orig_stdout = sys.stdout
    out_file = open(args.log_path, 'w')
    sys.stdout = out_file

    with open('/home/sana/multimodal/config.json', 'r') as f:
        configs = json.load(f)

    if not os.path.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    if not os.path.exists(args.plot_path):
        os.mkdir(args.plot_path)

    M1 = args.modality_names[0]
    M2 = args.modality_names[1]
    data_paths = {M1:configs[M1]["raw_data_path"], M2:configs[M2]["raw_data_path"]}
    
    # Load pretrain models.
    decoder_1, encoder_1, decoder_2, encoder_2 = load_pretrained_models(decoder_path_1=configs[M1]["decoder_path"], 
                                                                        encoder_path_1=configs[M1]["encoder_path"],
                                                                        decoder_path_2=configs[M2]["decoder_path"], 
                                                                        encoder_path_2=configs[M2]["encoder_path"])

    # Load data
    sample_list = get_paired_id_list(data_paths, from_file=False, file_name="/home/sana/multimodal/data_list.pkl")
    print("Total number of samples: ", len(sample_list))
    train_loader, _, _, list_ids = load_data(sample_list, n_train=args.n_samples, data_paths=data_paths, test_ratio=0.01)

    # Train the dissentangler 
    rep_disentangler = DeMuReFusion(encoders={M1: encoder_1, M2:encoder_2}, 
                                 decoders={M1: decoder_1, M2:decoder_2}, 
                                 train_ids=list_ids[0], shared_size=args.zs_size, modality_names=args.modality_names, 
                                 z_sizes={M1:args.zm_size, M2:args.zm_size},
                                 modality_shapes={M1:configs[M1]["input_size"], M2:configs[M2]["input_size"]},
                                 mask=True, beta=args.beta, gamma=args.gamma, ckpt_path=args.ckpt_path)  
    del decoder_1, encoder_1, encoder_2, decoder_2


    dec_loss, enc_loss, shared_loss, modality_loss = rep_disentangler.train(train_loader, epochs_enc=args.n_epochs, epochs_dec=args.n_epochs, 
                                                                            lr_dec=1e-3, lr_enc=1e-3, iteration_count=20,
                                                                            extra_encoder_training=20, no_mask_epochs=2
                                                                            )
    
    _, axs = plt.subplots(1,3, figsize=(12, 4))
    axs[0].plot(modality_loss)
    axs[0].set_title("modality-specific training loss")
    axs[1].plot(shared_loss)
    axs[1].set_title("shared training loss")
    axs[2].plot(enc_loss)
    axs[2].set_title("Encoder training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_path,"train_curves.pdf"))


if __name__=="__main__": 
    main()
