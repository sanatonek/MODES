import argparse
import json
import sys
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append('/home/sana/ml4h')
sys.path.append('/home/sana')
import tensorflow as tf
# from tensorflow_addons.activations import tfa_activations

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme
import pandas as pd
import seaborn as sns
import numpy as np
from tensorflow import keras 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from multimodal.representation import MultimodalRep
from multimodal.utils import *


parser = argparse.ArgumentParser(description='DeMuRe Fusion')
parser.add_argument('--log_path', default='/home/sana/multimodal/logs/eval_out.txt')
parser.add_argument('--plot_path', default='/home/sana/multimodal/plots/')
parser.add_argument('--ckpt_path', default='/home/sana/multimodal/ckpts_22/')
parser.add_argument('--zm_size', default=500)
parser.add_argument('--zs_size', default=700)
parser.add_argument('--beta', default=1e-6)
parser.add_argument('--gamma', default=1)
parser.add_argument('--n_samples', default=1000)
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



    orig_stdout = sys.stdout
    f = open('/home/sana/multimodal/logs/eval_ablation.txt', 'w')
    sys.stdout = f
    ckpt_nomask_path = "/home/sana/multimodal/ckpts_no_mask"

    z_s_size = 700#256
    z_m_size = 500#
    
    
    # Load pretrain models
    decoder_1, encoder_1, decoder_2, encoder_2 = load_pretrained_models(decoder_path_1=configs[M1]["decoder_path"], 
                                                                        encoder_path_1=configs[M1]["encoder_path"],
                                                                        decoder_path_2=configs[M2]["decoder_path"], 
                                                                        encoder_path_2=configs[M2]["encoder_path"],
                                                                        shared_s=args.zs_size, modality_specific_s=args.zm_size)
    
    # Load data
    sample_list = get_paired_id_list([data_paths[M1], data_paths[M2]], from_file=True, file_name="/home/sana/multimodal/data_list.pkl")
    print("Total number of samples: ", len(sample_list))
    _, valid_loader, test_loader, list_ids, trainset = load_data(sample_list, n_train=args.n_samples, data_paths=data_paths, test_ratio=0.01, get_trainset=True)

    # Train the dissentangler 
    rep_disentangler = MultimodalRep(encoders={M1: encoder_1, M2:encoder_2}, 
                                    decoders={M1: decoder_1, M2:decoder_2}, 
                                    train_ids=list_ids[0], shared_size=args.zs_size, modality_names=args.modality_names, 
                                    z_sizes={M1:args.zm_size, M2:args.zm_size},
                                    modality_shapes={M1:configs[M1]["input_size"], M2:configs[M2]["input_size"]},
                                    mask=True, beta=args.beta, gamma=args.gamma, ckpt_path=args.ckpt_path)  
    rep_disentangler.load_from_checkpoint()
    rep_disentangler._set_trainable_mask(trainable=False)

    rep_disentangler_no_mask = MultimodalRep(encoders={M1: encoder_1, M2:encoder_2}, 
                                    decoders={M1: decoder_1, M2:decoder_2}, 
                                    train_ids=list_ids[0], shared_size=args.zs_size, modality_names=args.modality_names, 
                                    z_sizes={M1:args.zm_size, M2:args.zm_size},
                                    modality_shapes={M1:configs[M1]["input_size"], M2:configs[M2]["input_size"]},
                                    mask=True, beta=args.beta, gamma=args.gamma, ckpt_path=ckpt_nomask_path) 
    rep_disentangler_no_mask.load_from_checkpoint()
    rep_disentangler_no_mask._set_trainable_mask(trainable=False)


    test_batch, test_ids = next(iter(test_loader))
    valid_batch, valid_ids = next(iter(valid_loader))
    test_ids = test_ids.tolist()
    valid_ids = valid_ids.tolist()
    z_s_test, z_m_test = rep_disentangler.encode(test_batch)
    z_merged_test = rep_disentangler.merge_representations(test_batch)
    z_s_test_nm, z_m_test_nm = rep_disentangler_no_mask.encode(test_batch)
    z_merged_test_nm = rep_disentangler_no_mask.merge_representations(test_batch)
    z_s_valid, z_m_valid = rep_disentangler.encode(valid_batch)
    z_merged_valid = rep_disentangler.merge_representations(valid_batch)
    z_s_valid_nm, z_m_valid_nm = rep_disentangler_no_mask.encode(valid_batch)
    z_merged_valid_nm = rep_disentangler_no_mask.merge_representations(valid_batch)

    print('********** Test df:', z_merged_test.shape)

    i = 0
    for data_batch, valid_id_batch in valid_loader:
        z_s_all, z_m_all = rep_disentangler.encode(data_batch)
        z_merged_valid = tf.concat([z_merged_valid, rep_disentangler.merge_representations(data_batch)], 0)
        z_s_all_nm, z_m_all_nm = rep_disentangler_no_mask.encode(data_batch)
        z_merged_valid_nm = tf.concat([z_merged_valid_nm, rep_disentangler_no_mask.merge_representations(data_batch)], 0)
        for key in z_s_all.keys():
            z_s_valid[key] = tf.concat([z_s_valid[key], z_s_all[key]], 0)
            z_m_valid[key] = tf.concat([z_m_valid[key], z_m_all[key]], 0)
            z_s_valid_nm[key] = tf.concat([z_s_valid_nm[key], z_s_all_nm[key]], 0)
            z_m_valid_nm[key] = tf.concat([z_m_valid_nm[key], z_m_all_nm[key]], 0)
        valid_ids.extend(valid_id_batch.tolist()) 
    print('********** Valid df: ', z_merged_valid.shape)

        
    ## Modality specific phenotypes
    ecg_pheno = ['PQInterval', 'QTInterval','QRSDuration','RRInterval']
    mri_pheno = ['LVEDV', 'LVESV', 'LVM', 'LVSV', 'RVEDV', 'RVESV', 'RVSV']
    shared_pheno = ['21003_Age-when-attended-assessment-centre_2_0', '21001_Body-mass-index-BMI_2_0']
    diagnostics = [ 'diabetes_type_2', 'valvular_disease_unspecified', 'mitral_regurgitation', 'Normal_sinus_rhythm', 'Sinus_bradycardia']
    phenotype_df = pd.read_csv("/home/sana/tensors_all_union.csv")[ecg_pheno+mri_pheno+shared_pheno+diagnostics+['fpath']]
    phenotype_df['fpath'] = phenotype_df['fpath'].astype(int)
    phenotype_df.dropna(inplace=True)
    zs_size = int(np.sum(rep_disentangler.shared_mask.binary_mask))
    rep_df = pd.DataFrame({'fpath': test_ids, 
                           'zm_ecg':z_m_test['input_ecg_rest_median_raw_10_continuous'].numpy().tolist() ,
                           'zm_mri':z_m_test['input_lax_4ch_heart_center_continuous'].numpy().tolist(),
                           'z_s': z_merged_test[:, -zs_size:].numpy().tolist(),
                           'z_merged': z_merged_test.numpy().tolist(), 
                           'zm_ecg (no mask)':z_m_test_nm['input_ecg_rest_median_raw_10_continuous'].numpy().tolist() ,
                           'zm_mri (no mask)':z_m_test_nm['input_lax_4ch_heart_center_continuous'].numpy().tolist(),
                           'z_s (no mask)': z_merged_test_nm[:, -zs_size:].numpy().tolist(),
                           'z_merged (no mask)': z_merged_test_nm.numpy().tolist()}) 
    rep_df_valid = pd.DataFrame({'fpath': valid_ids, 
                           'zm_ecg':z_m_valid['input_ecg_rest_median_raw_10_continuous'].numpy().tolist() ,
                           'zm_mri':z_m_valid['input_lax_4ch_heart_center_continuous'].numpy().tolist(),
                           'z_s': z_merged_valid[:, -zs_size:].numpy().tolist(),
                           'z_merged': z_merged_valid.numpy().tolist(),
                           'zm_ecg (no mask)':z_m_valid_nm['input_ecg_rest_median_raw_10_continuous'].numpy().tolist() ,
                           'zm_mri (no mask)':z_m_valid_nm['input_lax_4ch_heart_center_continuous'].numpy().tolist(),
                           'z_s (no mask)': z_merged_valid_nm[:, -zs_size:].numpy().tolist(),
                           'z_merged (no mask)': z_merged_valid_nm.numpy().tolist()})
    valid_pheno_df = pd.merge(phenotype_df, rep_df_valid, on='fpath')
    pheno_mean = valid_pheno_df[ecg_pheno+mri_pheno+shared_pheno].mean()
    pheno_std = valid_pheno_df[ecg_pheno+mri_pheno+shared_pheno].std()
    valid_pheno_df[ecg_pheno+mri_pheno+shared_pheno] = (valid_pheno_df[ecg_pheno+mri_pheno+shared_pheno]-pheno_mean)/pheno_std
    test_pheno_df = pd.merge(phenotype_df, rep_df, on='fpath')
    test_pheno_df[ecg_pheno+mri_pheno+shared_pheno] = (test_pheno_df[ecg_pheno+mri_pheno+shared_pheno]-pheno_mean)/pheno_std

    kfold_indices = []
    indices = list(range(len(valid_pheno_df)))
    for _ in range(3):
        random.shuffle(indices)
        train_index = indices[:int(0.6*len(valid_pheno_df))]
        test_index = indices[-int(0.6*len(valid_pheno_df)):]
        kfold_indices.append((train_index, test_index))
    # TODO: put this into a function
    # print("\nEvaluating modality-specific ECG")
    # s1 = phenotype_predictor(np.vstack(valid_pheno_df['zm_ecg']), valid_pheno_df, 
    #                          np.vstack(test_pheno_df['zm_ecg']), test_pheno_df,
    #                          phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, 
    #                          mask=rep_disentangler.modality_masks['input_ecg_rest_median_raw_10_continuous'].binary_mask)
    # s1_nm = phenotype_predictor(np.vstack(valid_pheno_df['zm_ecg (no mask)']), valid_pheno_df, 
    #                          np.vstack(test_pheno_df['zm_ecg (no mask)']), test_pheno_df,
    #                          phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    # print("\nEvaluating modality-specific MRI")
    # s2 = phenotype_predictor(np.vstack(valid_pheno_df['zm_mri']), valid_pheno_df, 
    #                          np.vstack(test_pheno_df['zm_mri']), test_pheno_df, 
    #                          phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, 
    #                          mask=rep_disentangler.modality_masks['input_lax_4ch_heart_center_continuous'].binary_mask)
    # s2_nm = phenotype_predictor(np.vstack(valid_pheno_df['zm_mri (no mask)']), valid_pheno_df, 
    #                          np.vstack(test_pheno_df['zm_mri (no mask)']), test_pheno_df, 
    #                          phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    # print("\nEvaluating shared merged")
    # s3 = phenotype_predictor(np.vstack(valid_pheno_df['z_s']), valid_pheno_df, 
    #                          np.vstack(test_pheno_df['z_s']), test_pheno_df, 
    #                          phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    # s3_nm = phenotype_predictor(np.vstack(valid_pheno_df['z_s (no mask)']), valid_pheno_df, 
    #                          np.vstack(test_pheno_df['z_s (no mask)']), test_pheno_df, 
    #                          phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    print("\nEvaluating merged representations")
    m1 = phenotype_predictor(np.vstack(valid_pheno_df['z_merged']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_merged']), test_pheno_df, kfold_indices=kfold_indices, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    m1_nm = phenotype_predictor(np.vstack(valid_pheno_df['z_merged (no mask)']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_merged (no mask)']), test_pheno_df, kfold_indices=kfold_indices, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)


    labels = ecg_pheno+mri_pheno
    x = np.arange(len(labels))
    width = 0.2  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained', figsize=(1.7*len(x), 4))
    scores = {'Masking':[np.mean(m1[pheno][1]) for pheno in labels],
            'No mask':[np.mean(m1_nm[pheno][1]) for pheno in labels],
            }
    scores_error = {'Masking':[np.std(m1[pheno][1]) for pheno in labels],
                    'No mask':[np.std(m1_nm[pheno][1]) for pheno in labels],
                    }
    for attribute, measurement in scores.items():
        offset = width * multiplier
        rounded_list = [round(m*100)/100 for m in measurement]
        rects = ax.bar(x + offset, rounded_list, width, label=attribute)
        ax.errorbar(x + offset, rounded_list, yerr=scores_error[attribute], fmt="o", color="black")
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('R-squared')
    ax.set_title('Predictive performance of diagnostic pheneyptes with and without mask', fontsize=14)
    ax.set_xticks(x + width, labels)#, rotation=40)
    ax.legend(loc='upper left', ncols=4)
    plt.savefig(os.path.join(args.plot_path,"mask_no_mask_prediction.pdf"))
    fig.clf()

    labels = diagnostics
    diagnostics_ticks = ['Diabetes (type 2)', 'Valvular disease', 'Mitral regurgitation', 'Normal rhythm', 'Sinus bradycardia']
    x = np.arange(len(labels))
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained', figsize=(1.7*len(x), 4))
    scores = {'Masking':[np.mean(m1[pheno][1]) for pheno in labels],
            'No mask':[np.mean(m1_nm[pheno][1]) for pheno in labels],
            }
    scores_error = {'Masking':[np.std(m1[pheno][1]) for pheno in labels],
                    'No mask':[np.std(m1_nm[pheno][1]) for pheno in labels],
                    }
    for attribute, measurement in scores.items():
        offset = width * multiplier
        rounded_list = [round(m*100)/100 for m in measurement]
        rects = ax.bar(x + offset, rounded_list, width, label=attribute)
        ax.errorbar(x + offset, rounded_list, yerr=scores_error[attribute], fmt="o", color="black")
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('AUROC')
    ax.set_title('Predictive performance of diagnostic labels with and without mask', fontsize=14)
    ax.set_xticks(x + width, diagnostics_ticks, fontsize=9)#, rotation=40)
    ax.legend(loc='upper left', ncols=4)
    plt.savefig(os.path.join(args.plot_path,"mask_no_mask_prediction_diag.pdf"))
    fig.clf()

 

    sys.stdout = orig_stdout
    f.close()


if __name__=="__main__":
    # with open('/home/sana/multimodal/logs/out.txt', 'w') as sys.stdout:
    main()
