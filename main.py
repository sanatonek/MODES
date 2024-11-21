
import sys
import os
import numpy as np
sys.path.append('/home/sana/ml4h')
sys.path.append('/home/sana')
import tensorflow as tf
from tensorflow import keras 

import matplotlib.pyplot as plt
import random
import seaborn as sns
sns.set_theme

from multimodal.representation import MultimodalRep
from multimodal.utils import load_pretrained_models, load_data, get_id_list, plot_sample, pca_analysis, get_ref_sample

tf.keras.callbacks.TerminateOnNaN()
tf.config.list_physical_devices('GPU')

def main():
    orig_stdout = sys.stdout
    out_file = open('/home/sana/multimodal/logs/out.txt', 'w')
    sys.stdout = out_file

    data_path = "/mnt/disks/google-annotated-cardiac-tensors-45k-2021-03-25/2020-09-21"
    ecg_decoder_path = '/home/sana/model_ckpts/decoder_ecg_rest_median_raw_10.h5'
    ecg_encoder_path = '/home/sana/model_ckpts/encoder_ecg_rest_median_raw_10.h5'
    mri_encoder_path = '/home/sana/model_ckpts/encoder_lax_4ch_heart_center.h5'
    mri_decoder_path = '/home/sana/model_ckpts/decoder_lax_4ch_heart_center.h5'
    modality_names = ['input_ecg_rest_median_raw_10_continuous', 'input_lax_4ch_heart_center_continuous']
    data_paths = {'input_lax_4ch_heart_center_continuous':data_path, 'input_ecg_rest_median_raw_10_continuous':data_path}
    ckpt_path = "/home/sana/multimodal/ckpts"
    plot_path = "/home/sana/multimodal/plots"

    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    z_s_size = 512#1024
    z_m_size = 256#512
    
    # Load pretrain models.
    ecg_decoder, ecg_encoder, mri_encoder, mri_decoder = load_pretrained_models(ecg_decoder_path, ecg_encoder_path, mri_encoder_path, mri_decoder_path,
                                                                                shared_s=z_s_size, modality_specific_s=z_m_size)

    # Load data
    sample_list = get_id_list(data_path, from_file=True, file_name="/home/sana/multimodal/data_list.pkl")
    # sample_list = sample_list
    print("Total number of samples: ", len(sample_list))
    train_loader, _, _, list_ids = load_data(sample_list, n_train=5000, data_paths=data_paths, train_ratio=0.9, test_ratio=0.05)
    n_train = int(len(sample_list)*0.6)
    ref_samples = get_ref_sample(data_path, list_ids[0][10])

    # Train the dissentangler 
    rep_disentangler = MultimodalRep(encoders={'input_ecg_rest_median_raw_10_continuous': ecg_encoder, 'input_lax_4ch_heart_center_continuous':mri_encoder}, 
                                 decoders={'input_ecg_rest_median_raw_10_continuous': ecg_decoder, 'input_lax_4ch_heart_center_continuous':mri_decoder}, 
                                 train_ids=list_ids[0], shared_size=z_s_size, modality_names=modality_names, 
                                 z_sizes={'input_ecg_rest_median_raw_10_continuous':z_m_size, 'input_lax_4ch_heart_center_continuous':z_m_size},
                                 modality_shapes={'input_ecg_rest_median_raw_10_continuous':(600, 12), 'input_lax_4ch_heart_center_continuous':(96, 96, 50)},
                                 mask=True, beta=0.0001, gamma=0.01, ckpt_path=ckpt_path)  
    
    del ecg_decoder, ecg_encoder, mri_encoder, mri_decoder

    dec_loss, enc_loss, shared_loss, modality_loss = rep_disentangler.train(train_loader, epochs_enc=5, epochs_dec=5, lr_dec=1e-3, lr_enc=1e-3, iteration_count=20)
    
    _, axs = plt.subplots(1,3, figsize=(12, 4))
    # axs[0, 0].plot(dec_loss)
    # axs[0, 0].set_title("Decoder training loss")
    axs[0].plot(modality_loss)
    axs[0].set_title("modality-specific training loss")
    axs[1].plot(shared_loss)
    axs[1].set_title("shared training loss")
    axs[2].plot(enc_loss)
    axs[2].set_title("Encoder training loss")
    plt.tight_layout()
    plt.savefig("/home/sana/multimodal/plots/train_curves.pdf")


if __name__=="__main__":
    # with open('/home/sana/multimodal/logs/out.txt', 'w') as sys.stdout:
    main()
