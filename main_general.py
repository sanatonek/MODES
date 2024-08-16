
import sys
import os
import numpy as np
sys.path.append('/home/sana/ml4h')
sys.path.append('/home/sana')
import tensorflow as tf

import matplotlib.pyplot as plt
import random
import seaborn as sns
sns.set_theme

from multimodal.contrastive import MultimodalRep
from multimodal.utils import load_pretrained_models, load_data_affect, load_data, get_id_list, plot_sample, pca_analysis, get_ref_sample

tf.keras.callbacks.TerminateOnNaN()
tf.config.list_physical_devices('GPU')

def main():
    orig_stdout = sys.stdout
    out_file = open('/home/sana/multimodal/logs/out.txt', 'w')
    sys.stdout = out_file

    # load_data_affect()

    data_path = "/mnt/disks/google-annotated-cardiac-tensors-45k-2021-03-25/2020-09-21"
    # ecg_decoder_path = '/home/sana/ml4h/model_zoo/dropfuse/decoder_ecg_rest_median_raw_10.h5'
    # ecg_encoder_path = '/home/sana/ml4h/model_zoo/dropfuse/encoder_ecg_rest_median_raw_10.h5'
    # mri_encoder_path = '/home/sana/ml4h/model_zoo/dropfuse/encoder_lax_4ch_heart_center.h5'
    # mri_decoder_path = '/home/sana/ml4h/model_zoo/dropfuse/decoder_lax_4ch_heart_center.h5'
    ecg_decoder_path = '/home/sana/model_ckpts/decoder_ecg_rest_median_raw_10.h5'
    ecg_encoder_path = '/home/sana/model_ckpts/encoder_ecg_rest_median_raw_10.h5'
    mri_encoder_path = '/home/sana/model_ckpts/encoder_lax_4ch_heart_center.h5'
    mri_decoder_path = '/home/sana/model_ckpts/decoder_lax_4ch_heart_center.h5'
    modality_names = ['input_ecg_rest_median_raw_10_continuous', 'input_lax_4ch_heart_center_continuous']

    if not os.path.exists("/home/sana/multimodal/ckpts"):
        os.mkdir("/home/sana/multimodal/ckpts")
    if not os.path.exists("/home/sana/multimodal/plots"):
        os.mkdir("/home/sana/multimodal/plots")

    z_s_size = 256
    z_m_size = 64
    
    # Load pretrain models
    ecg_decoder, ecg_encoder, mri_encoder, mri_decoder = load_pretrained_models(ecg_decoder_path, ecg_encoder_path, mri_encoder_path, mri_decoder_path,
                                                                                shared_s=z_s_size, modality_specific_s=z_m_size)

    # Load data
    sample_list = get_id_list(data_path, from_file=False, file_name="/home/sana/multimodal/data_list.pkl")
    sample_list = sample_list
    print("Total number of samples: ", len(sample_list))
    train_loader, valid_loader, test_loader, list_ids = load_data(sample_list, data_path, train_ratio=0.9, test_ratio=0.05)
    n_train = int(len(sample_list)*0.6)
    ref_samples = get_ref_sample(data_path, list_ids[0][10])

    # Train the dissentangler 
    rep_disentangler = MultimodalRep(encoders={'input_ecg_rest_median_raw_10_continuous': ecg_encoder, 'input_lax_4ch_heart_center_continuous':mri_encoder}, 
                                 n_train=len(list_ids[0]), shared_size=z_s_size, modality_names=modality_names, 
                                 z_sizes={'input_ecg_rest_median_raw_10_continuous':z_m_size, 'input_lax_4ch_heart_center_continuous':z_m_size},
                                 modality_shapes={'input_ecg_rest_median_raw_10_continuous':(600, 12), 'input_lax_4ch_heart_center_continuous':(96, 96, 50)},
                                 mask=True, beta=1.0)  
    
    del ecg_decoder, ecg_encoder, mri_encoder, mri_decoder

    dec_loss, enc_loss, shared_loss, modality_loss = rep_disentangler.train(train_loader, epochs_enc=4, epochs_dec=4, lr_dec=0.001, lr_enc=0.001)
    _, axs = plt.subplots(2,2)
    axs[0, 0].plot(dec_loss)
    axs[0, 0].set_title("Decoder training loss")
    axs[0, 1].plot(enc_loss)
    axs[0, 1].set_title("Encoder training loss")
    axs[1, 0].plot(shared_loss)
    axs[1, 0].set_title("shared training loss")
    axs[1, 1].plot(modality_loss)
    axs[1, 1].set_title("modality-specific training loss")
    plt.tight_layout()
    plt.savefig("/home/sana/multimodal/plots/train_curves.pdf")

    rep_disentangler.load_from_checkpoint("/home/sana/multimodal/ckpts")

    # Plot reconstructed samples
    test_batch, _ = next(iter(test_loader))
    z_s_all, z_m_all = rep_disentangler.encode(test_batch)
    reconstructed_samples = rep_disentangler.decode(z_s_all, z_m_all)#, test_batch)
    z_s_mixed = {}
    z_s_mixed[modality_names[0]] = z_s_all[modality_names[1]]
    z_s_mixed[modality_names[1]] = z_s_all[modality_names[0]]
    reconstructed_mixed_samples = rep_disentangler.decode(z_s_mixed, z_m_all)#, test_batch)
    for m_name in modality_names:
        plot_sample(test_batch[m_name], num_cols=4, num_rows=1,
                    save_path="/home/sana/multimodal/plots/original_%s.pdf"%m_name)
        plot_sample(reconstructed_samples[m_name], num_cols=4, num_rows=1,
                    save_path="/home/sana/multimodal/plots/reconstructed_%s.pdf"%m_name)
        plot_sample(reconstructed_mixed_samples[m_name], num_cols=4, num_rows=1,
                    save_path="/home/sana/multimodal/plots/reconstructed_mixed_%s.pdf"%m_name)
    sys.stdout = orig_stdout
    out_file.close()


if __name__=="__main__":
    # with open('/home/sana/multimodal/logs/out.txt', 'w') as sys.stdout:
    main()