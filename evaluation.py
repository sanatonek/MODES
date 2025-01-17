
import sys
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append('/home/sana/ml4h')
sys.path.append('/home/sana')
import tensorflow as tf
import json
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

from multimodal.fusion import DeMuReFusion
from multimodal.utils import *


parser = argparse.ArgumentParser(description='DeMuRe Fusion')
parser.add_argument('--log_path', default='/home/sana/multimodal/logs/eval_out.txt')
parser.add_argument('--plot_path', default='/home/sana/multimodal/plots/')
parser.add_argument('--ckpt_path', default='/home/sana/multimodal/ckpts_25/')
parser.add_argument('--zm_size', default=400)
parser.add_argument('--zs_size', default=600)
parser.add_argument('--beta', default=1e-6)
parser.add_argument('--gamma', default=1)
parser.add_argument('--n_samples', default=1000)
parser.add_argument('--modality_names', nargs='+', default=['input_ecg_rest_median_raw_10_continuous', 'input_lax_4ch_heart_center_continuous'])



def main():
    args = parser.parse_args()
    orig_stdout = sys.stdout
    out_file = open(args.log_path, 'w')
    sys.stdout = out_file

    ecg_df_encoder_path = '/home/sana/model_ckpts/encoder_ecg_rest_median_raw_10_1k.h5'
    mri_df_encoder_path = '/home/sana/model_ckpts/encoder_lax_4ch_heart_center_1k.h5'

    with open('/home/sana/multimodal/config.json', 'r') as f:
        configs = json.load(f)

    if not os.path.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    if not os.path.exists(args.plot_path):
        os.mkdir(args.plot_path)

    M1 = args.modality_names[0]
    M2 = args.modality_names[1]
    data_paths = {M1:configs[M1]["raw_data_path"], M2:configs[M2]["raw_data_path"]}
    
    ## Load pretrain models.
    decoder_1, encoder_1, decoder_2, encoder_2 = load_pretrained_models(decoder_path_1=configs[M1]["decoder_path"], 
                                                                        encoder_path_1=configs[M1]["encoder_path"],
                                                                        decoder_path_2=configs[M2]["decoder_path"], 
                                                                        encoder_path_2=configs[M2]["encoder_path"],
                                                                        shared_s=args.zs_size, modality_specific_s=args.zm_size)
    
    encoder_1.trainable = False
    encoder_2.trainable = False
    ecg_df_encoder = tf.keras.models.load_model(ecg_df_encoder_path)
    mri_df_encoder = tf.keras.models.load_model(mri_df_encoder_path)
    ecg_df_encoder.trainable = False
    mri_df_encoder.trainable = False
    
    ## Load data
    sample_list = get_paired_id_list(data_paths, from_file=False, file_name="/home/sana/multimodal/data_list.pkl")
    print("Total number of samples: ", len(sample_list))
    _, valid_loader, test_loader, list_ids, trainset = load_data(sample_list, n_train=args.n_samples, data_paths=data_paths, test_ratio=0.01, get_trainset=True)

    ## Train the dissentangler 
    rep_disentangler = DeMuReFusion(encoders={M1: encoder_1, M2:encoder_2}, 
                                 decoders={M1: decoder_1, M2:decoder_2}, 
                                 train_ids=list_ids[0], shared_size=args.zs_size, modality_names=args.modality_names, 
                                 z_sizes={M1:args.zm_size, M2:args.zm_size},
                                 modality_shapes={M1:configs[M1]["input_size"], M2:configs[M2]["input_size"]},
                                 mask=True, beta=args.beta, gamma=args.gamma, ckpt_path=args.ckpt_path)  
    rep_disentangler.load_from_checkpoint()
    rep_disentangler._set_trainable_mask(trainable=False)


    ## Plot training analysis
    train_tracker = np.load(os.path.join(args.ckpt_path,'train_tracker.npz'))
    _, axs = plt.subplots(len(args.modality_names)+1, 1, figsize=(6,8))
    sns.heatmap(np.stack(train_tracker['shared_mask']), ax=axs[2])
    sns.heatmap(train_tracker['m1_mask'], ax=axs[0])
    sns.heatmap(train_tracker['m2_mask'], ax=axs[1])
    axs[0].set_ylabel('Epochs')
    axs[0].set_title("ECG masks")
    axs[1].set_ylabel('Epochs')
    axs[1].set_title("Cardiac MRI masks")
    axs[2].set_ylabel('Epochs')
    axs[2].set_title("Shared masks")
    axs[2].set_xlabel("Dimensions", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_path,"masks.pdf"))

    _, axs = plt.subplots(1,3, figsize=(12, 4))
    axs[0].plot(np.where(train_tracker['modality_loss']<50, train_tracker['modality_loss'], 50*np.zeros_like(train_tracker['modality_loss'])))
    axs[0].set_title("modality-specific training loss", fontsize=14)
    axs[0].set_xlabel("Epochs", fontsize=14)
    axs[1].plot(np.where(train_tracker['shared_loss']<50, train_tracker['shared_loss'], 50*np.zeros_like(train_tracker['shared_loss'])))
    axs[1].set_title("shared training loss", fontsize=14)
    axs[1].set_xlabel("Epochs", fontsize=14)
    axs[2].plot(np.where(train_tracker['encoder_loss']<50, train_tracker['encoder_loss'], 50*np.zeros_like(train_tracker['encoder_loss'])))
    axs[2].set_title("Encoder training loss", fontsize=14)
    axs[2].set_xlabel("Epochs", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_path,"train_curves.pdf"))


    test_batch, test_ids = next(iter(test_loader))
    valid_batch, valid_ids = next(iter(valid_loader))
    test_ids = test_ids.tolist()
    valid_ids = valid_ids.tolist()
    z_s_test, z_m_test = rep_disentangler.encode(test_batch)
    z_merged_test = rep_disentangler.merge_representations(test_batch)
    z_s_valid, z_m_valid = rep_disentangler.encode(valid_batch)
    z_merged_valid = rep_disentangler.merge_representations(valid_batch)
    baseline_z_test = {M1:encoder_1(tf.convert_to_tensor(test_batch[M1])), 
                       M2:encoder_2(tf.convert_to_tensor(test_batch[M2]))}
    baseline_z_valid = {M1:encoder_1(tf.convert_to_tensor(valid_batch[M1])), 
                       M2:encoder_2(tf.convert_to_tensor(valid_batch[M2]))}
    dropfuse_z_valid = {M1:ecg_df_encoder((tf.convert_to_tensor(valid_batch[M1]))), 
                       M2:mri_df_encoder((tf.convert_to_tensor(valid_batch[M2])))}
    dropfuse_z_test = {M1:ecg_df_encoder((tf.convert_to_tensor(test_batch[M1]))), 
                       M2:mri_df_encoder((tf.convert_to_tensor(test_batch[M2])))}

    for data_batch, valid_id_batch in valid_loader:
        z_s_all, z_m_all = rep_disentangler.encode(data_batch)
        z_merged_valid = tf.concat([z_merged_valid, rep_disentangler.merge_representations(data_batch)], 0)
        for key in z_s_all.keys():
            z_s_valid[key] = tf.concat([z_s_valid[key], z_s_all[key]], 0)
            z_m_valid[key] = tf.concat([z_m_valid[key], z_m_all[key]], 0)
            if 'lax' in key:
                baseline_z_valid[key] = tf.concat([baseline_z_valid[key], encoder_2(tf.convert_to_tensor(data_batch[key]))], 0)
                dropfuse_z_valid[key] = tf.concat([dropfuse_z_valid[key], mri_df_encoder((tf.convert_to_tensor(data_batch[key])))], 0)
            if 'ecg' in key:
                baseline_z_valid[key] = tf.concat([baseline_z_valid[key], encoder_1(tf.convert_to_tensor(data_batch[key]))], 0)
                dropfuse_z_valid[key] = tf.concat([dropfuse_z_valid[key], ecg_df_encoder((tf.convert_to_tensor(data_batch[key])))], 0)
        valid_ids.extend(valid_id_batch.tolist()) 
    print('********** Valid df: ', z_merged_valid.shape)
 
    # plot generated samples
    rnd_id = np.random.randint(0,len(test_batch))
    generated_M2s, generated_M1s = [], []
    M2_pcas, M1_pcas = [], []
    for i in range(3):
        generated_M2, max_pc_mri_inds, min_pc_mri_inds = (rep_disentangler.generate_sample(test_batch, rnd_id, n_pca=i, variation=True,
                                                            ref_modality=M1, target_modality=M2, n_samples=3))
        generated_M1, max_pc_ecg_inds, min_pc_ecg_inds = (rep_disentangler.generate_sample(test_batch, rnd_id, n_pca=i, variation=True,
                                                            ref_modality=M2, target_modality=M1, n_samples=3))
        generated_M2s.append(generated_M2)
        generated_M1s.append(generated_M1)
        M2_pcas.extend([trainset[ind][0][M2] for ind in (min_pc_mri_inds+max_pc_mri_inds)])
        M1_pcas.extend([trainset[ind][0][M1] for ind in (min_pc_ecg_inds+max_pc_ecg_inds)])
    plot_sample(tf.concat(generated_M2s,0), num_cols=len(generated_M2s[0]), num_rows=3, 
                save_path=os.path.join(args.plot_path,"generated_distribution_mri.pdf"))
    plot_sample(tf.concat(generated_M1s,0), num_cols=len(generated_M2s[0]), num_rows=3, 
                save_path=os.path.join(args.plot_path,"generated_distribution_ecg.pdf"))
    plot_sample(M2_pcas, num_cols=len(generated_M2s[0]), num_rows=3, 
                save_path=os.path.join(args.plot_path,"pca_mri.pdf"))
    plot_sample(M1_pcas, num_cols=len(generated_M2s[0]), num_rows=3, 
                save_path=os.path.join(args.plot_path,"pca_ecg.pdf"))
    
    generated_mod, probs, ref_sample = rep_disentangler.generate_sample(test_batch, rnd_id, variation=False,
                                    ref_modality=M1, target_modality=M2, n_samples=8)
    plot_sample([ref_sample]+[g for g in generated_mod], num_cols=8+1, num_rows=1, 
                save_path=os.path.join("generated_similar_mri.pdf"))
    plt.bar(np.arange(8), probs)
    plt.title("Probability of generated samples")
    plt.savefig(os.path.join(args.plot_path,"generated_similar_mri_probs.pdf"))
    
    generated_mod, probs, ref_sample = rep_disentangler.generate_sample(test_batch, rnd_id, variation=False,
                                    ref_modality=M2, target_modality=M1, n_samples=8)
    plot_sample([ref_sample]+[g for g in generated_mod], num_cols=8+1, num_rows=1, 
                save_path=os.path.join(args.plot_path,"generated_similar_ecg.pdf"))

    reconstructed_samples = rep_disentangler.decode(z_s_test, z_m_test)
    z_s_mixed = {}
    z_s_mixed[M1] = z_s_test[M2]
    z_s_mixed[M2] = z_s_test[M1]
    reconstructed_mixed_samples = rep_disentangler.decode(z_s_mixed, z_m_test)
    for m_name in args.modality_names:
        plot_sample(test_batch[m_name], num_cols=4, num_rows=1,
                    save_path=os.path.join(args.plot_path,"original_%s.pdf"%m_name))
        plot_sample(reconstructed_samples[m_name], num_cols=4, num_rows=1,
                    save_path=os.path.join(args.plot_path,"reconstructed_%s.pdf"%m_name))
        plot_sample(reconstructed_mixed_samples[m_name], num_cols=4, num_rows=1,
                    save_path=os.path.join(args.plot_path,"reconstructed_mixed_%s.pdf"%m_name))
        
        # plot correlations
        corr_mat = np.corrcoef(np.concatenate([z_m_test[m_name], z_s_test[m_name]], axis=-1).T)
        plt.matshow(corr_mat)
        plt.savefig(os.path.join(args.plot_path,"%s_correlations.pdf"%m_name))
        plt.close()
        
    ## Modality specific phenotypes
    ecg_pheno = ['PQInterval', 'QTInterval','QRSDuration','RRInterval']
    mri_pheno = ['LVEDV', 'LVESV', 'LVM', 'LVSV', 'RVEDV', 'RVESV', 'RVSV']
    shared_pheno = ['21003_Age-when-attended-assessment-centre_2_0', '21001_Body-mass-index-BMI_2_0']
    diagnostics = [ 'diabetes_type_2', 'valvular_disease_unspecified', 'Normal_sinus_rhythm', 'Sinus_bradycardia']
    phenotype_df = pd.read_csv("/home/sana/tensors_all_union.csv")[ecg_pheno+mri_pheno+shared_pheno+diagnostics+['fpath']]
    phenotype_df['fpath'] = phenotype_df['fpath'].astype(int)
    phenotype_df.dropna(inplace=True)
    zs_size = int(np.sum(rep_disentangler.shared_mask.binary_mask))
    rep_df = pd.DataFrame({'fpath': test_ids, 
                           'zs_ecg':z_s_test[M1].numpy().tolist(),
                           'zs_mri':z_s_test[M2].numpy().tolist() ,
                           'zm_ecg':z_m_test[M1].numpy().tolist() ,
                           'zm_mri':z_m_test[M2].numpy().tolist(),
                           'z_s': z_merged_test[:, -zs_size:].numpy().tolist(),
                           'z_baseline_ecg': baseline_z_test[M1].numpy().tolist(),
                           'z_baseline_mri': baseline_z_test[M2].numpy().tolist(),
                           'z_dropfuse_ecg': dropfuse_z_test[M1].numpy().tolist(),
                           'z_dropfuse_mri': dropfuse_z_test[M2].numpy().tolist(),
                           'z_merged': z_merged_test.numpy().tolist()}) 
    rep_df_valid = pd.DataFrame({'fpath': valid_ids, 
                           'zs_ecg':z_s_valid[M1].numpy().tolist(),
                           'zs_mri':z_s_valid[M2].numpy().tolist() ,
                           'zm_ecg':z_m_valid[M1].numpy().tolist() ,
                           'zm_mri':z_m_valid[M2].numpy().tolist(),
                           'z_s': z_merged_valid[:, -zs_size:].numpy().tolist(),
                            'z_baseline_ecg': baseline_z_valid[M1].numpy().tolist(),
                            'z_baseline_mri': baseline_z_valid[M2].numpy().tolist(),
                            'z_dropfuse_ecg': dropfuse_z_valid[M1].numpy().tolist(),
                            'z_dropfuse_mri': dropfuse_z_valid[M2].numpy().tolist(),
                            'z_merged': z_merged_valid.numpy().tolist()})
    valid_pheno_df = pd.merge(phenotype_df, rep_df_valid, on='fpath')
    test_pheno_df = pd.merge(phenotype_df, rep_df, on='fpath')
    
    kfold_indices = []
    indices = list(range(len(valid_pheno_df)))
    for _ in range(3):
        random.shuffle(indices)
        train_index = indices[:int(0.6*len(valid_pheno_df))]
        test_index = indices[-int(0.6*len(valid_pheno_df)):]
        kfold_indices.append((train_index, test_index))
    print("\nEvaluating modality-specific ECG")
    s3 = phenotype_predictor(np.vstack(valid_pheno_df['zm_ecg']), valid_pheno_df, 
                             np.vstack(test_pheno_df['zm_ecg']), test_pheno_df,
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, kfold_indices=kfold_indices,
                             mask=rep_disentangler.modality_masks['input_ecg_rest_median_raw_10_continuous'].binary_mask)
    print("\nEvaluating modality-specific MRI")
    s4 = phenotype_predictor(np.vstack(valid_pheno_df['zm_mri']), valid_pheno_df, 
                             np.vstack(test_pheno_df['zm_mri']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, kfold_indices=kfold_indices,
                             mask=rep_disentangler.modality_masks['input_lax_4ch_heart_center_continuous'].binary_mask)
    print("\nEvaluating shared merged")
    s5 = phenotype_predictor(np.vstack(valid_pheno_df['z_s']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_s']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, kfold_indices=kfold_indices,)
    print("\nEvaluating full ECG")
    f1 = phenotype_predictor(np.concatenate([np.vstack(valid_pheno_df['zm_ecg']),np.vstack(valid_pheno_df['zs_ecg'])],-1),
                             valid_pheno_df,
                             np.concatenate([np.vstack(test_pheno_df['zm_ecg']),np.vstack(test_pheno_df['zs_ecg'])],-1), 
                             test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, kfold_indices=kfold_indices,
                             mask=np.concatenate([rep_disentangler.modality_masks['input_ecg_rest_median_raw_10_continuous'].binary_mask, rep_disentangler.shared_mask.binary_mask], 0)) 
    print("\nEvaluating full MRI")
    f2 = phenotype_predictor(np.concatenate([np.vstack(valid_pheno_df['zm_mri']),np.vstack(valid_pheno_df['zs_mri'])],-1),
                             valid_pheno_df,
                             np.concatenate([np.vstack(test_pheno_df['zm_mri']),np.vstack(test_pheno_df['zs_mri'])],-1), 
                             test_pheno_df,phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, kfold_indices=kfold_indices,
                             mask=np.concatenate([rep_disentangler.modality_masks['input_lax_4ch_heart_center_continuous'].binary_mask, rep_disentangler.shared_mask.binary_mask], 0))
    print("\nEvaluating baseline ECG")
    b1 = phenotype_predictor(np.vstack(valid_pheno_df['z_baseline_ecg']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_baseline_ecg']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, kfold_indices=kfold_indices,)
    print("\nEvaluating baseline MRI")
    b2 = phenotype_predictor(np.vstack(valid_pheno_df['z_baseline_mri']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_baseline_mri']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, kfold_indices=kfold_indices,)

    late_fusion = phenotype_predictor_lf(np.vstack(valid_pheno_df['z_baseline_ecg']), valid_pheno_df, 
                                         np.vstack(valid_pheno_df['z_baseline_mri']), valid_pheno_df, 
                                        phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, kfold_indices=kfold_indices,)
    print("\nEvaluating dropfuse ECG")
    df1 = phenotype_predictor(np.vstack(valid_pheno_df['z_dropfuse_ecg']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_dropfuse_ecg']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, kfold_indices=kfold_indices,)
    print("\nEvaluating dropfuse MRI")
    df2 = phenotype_predictor(np.vstack(valid_pheno_df['z_dropfuse_mri']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_dropfuse_mri']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, kfold_indices=kfold_indices,)
    print("\nEvaluating merged representations")
    m1 = phenotype_predictor(np.vstack(valid_pheno_df['z_merged']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_merged']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, kfold_indices=kfold_indices,)
    print("\nEvaluating concatenated baselines")
    m2 = phenotype_predictor(np.concatenate([np.vstack(valid_pheno_df['z_baseline_mri']),np.vstack(valid_pheno_df['z_baseline_ecg'])],-1), 
                             valid_pheno_df, 
                             np.concatenate([np.vstack(test_pheno_df['z_baseline_mri']),np.vstack(test_pheno_df['z_baseline_ecg'])],-1), 
                             test_pheno_df, kfold_indices=kfold_indices,
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    
    plot_pheno_prediction_performance(phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, 
                                      perf_results=[b1, b2, m1, m2], 
                                      labels=['ecg rep.', 'mri rep.', 'merged rep.', 'concat. re.'],
                                      tag='overall_perf')

    
    ## predictive performance test
    labels = ecg_pheno+mri_pheno
    x = np.arange(len(labels))
    width = 0.15  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained', figsize=(len(x)*2, 4))
    scores = {'Ours':[np.mean(m1[pheno][1]) for pheno in labels],
            'Early fusion':[np.mean(m2[pheno][1]) for pheno in labels],
            'Late fusion':[np.mean(late_fusion[pheno][1]) for pheno in labels],
            'Unimodal ECG':[np.mean(b1[pheno][1]) for pheno in labels],
            'Unimodal MRI':[np.mean(b2[pheno][1]) for pheno in labels],
            }
    scores_error = {'Ours':[np.std(m1[pheno][1]) for pheno in labels],
                    'Early fusion':[np.std(m2[pheno][1]) for pheno in labels],
                    'Late fusion':[np.std(late_fusion[pheno][1]) for pheno in labels],
                    'Unimodal ECG':[np.std(b1[pheno][1]) for pheno in labels],
                    'Unimodal MRI':[np.std(b2[pheno][1]) for pheno in labels],
                    }
    for attribute, measurement in scores.items():
        offset = width * multiplier
        rounded_list = [round(m*100)/100 for m in measurement]
        rects = ax.bar(x + offset, rounded_list, width, label=attribute)
        ax.errorbar(x + offset, rounded_list, yerr=scores_error[attribute], fmt="o", color="black")
        multiplier += 1
    ax.set_ylabel('R-squared')
    ax.set_title('Performance of representations for predicting diagnostic phenotypes', fontsize=14)
    ax.set_xticks(x + width, labels)
    ax.legend(loc='upper right', ncols=5)
    plt.savefig(os.path.join(args.plot_path,"phenotype_prediction.pdf"))
    fig.clf()
    print("Mean predictive performances: ", [np.mean(m1[pheno][1]) for pheno in labels])


    diagnostics_ticks = ['Diabetes (type 2)', 'Valvular disease', 'Normal rhythm', 'Sinus bradycardia']

    labels = diagnostics
    x = np.arange(len(labels))
    width = 0.15  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained', figsize=(3*len(x), 4))
    scores = {'Multimodal (Ours)':[np.mean(m1[pheno][1]) for pheno in labels],
            'Multimodal (Early fusion)':[np.mean(m2[pheno][1]) for pheno in labels],
            'Multimodal (Late fusion)':[np.mean(late_fusion[pheno][1]) for pheno in labels],
            'Unimodal ECG':[np.mean(b1[pheno][1]) for pheno in labels],
            'Unimodal MRI':[np.mean(b2[pheno][1]) for pheno in labels],
            }
    scores_error = {'Multimodal (Ours)':[np.std(m1[pheno][1]) for pheno in labels],
                    'Multimodal (Early fusion)':[np.std(m2[pheno][1]) for pheno in labels],
                    'Multimodal (Late fusion)':[np.std(late_fusion[pheno][1]) for pheno in labels],
                    'Unimodal ECG':[np.std(b1[pheno][1]) for pheno in labels],
                    'Unimodal MRI':[np.std(b2[pheno][1]) for pheno in labels],
                    }
    for attribute, measurement in scores.items():
        offset = width * multiplier
        rounded_list = [round(m*100)/100 for m in measurement]
        rects = ax.bar(x + offset, rounded_list, width, label=attribute)
        ax.errorbar(x + offset, rounded_list, yerr=scores_error[attribute], fmt="o", color="black")
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('AUROC')
    ax.set_ylim(0, 1)
    ax.set_title('Performance of representations for predicting diagnostic labels', fontsize=14)
    ax.set_xticks(x + width, diagnostics_ticks, fontsize=9)
    ax.legend(loc='upper left', ncols=5)
    plt.savefig(os.path.join(args.plot_path,"diagnosis_prediction.pdf"))
    fig.clf()


    
    # Missing modality experiment
    width = 0.2  # the width of the bars
    multiplier = 0
    x = np.arange(len(ecg_pheno))
    fig, ax = plt.subplots(layout='constrained', figsize=(len(x)*2, 4))
    scores = {'Our MRI':[np.mean(f2[pheno][1]) for pheno in ecg_pheno],
            'Unimodal MRI':[np.mean(b2[pheno][1]) for pheno in ecg_pheno],
            'DropFuse MRI':[np.mean(df2[pheno][1]) for pheno in ecg_pheno],
            }
    scores_error = {'Our MRI':[np.std(f2[pheno][1]) for pheno in ecg_pheno],
                    'Unimodal MRI':[np.std(b2[pheno][1]) for pheno in ecg_pheno],
                    'DropFuse MRI':[np.std(df2[pheno][1]) for pheno in ecg_pheno],
                    }
    for attribute, measurement in scores.items():
        offset = width * multiplier
        rounded_list = [round(m*100)/100 for m in measurement]
        rects = ax.bar(x + offset, rounded_list, width, label=attribute)
        ax.errorbar(x + offset, rounded_list, yerr=scores_error[attribute], fmt="o", color="black")
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('R-squared')
    ax.set_title('Prediction of ECG derived phenotypes from MRI', fontsize=14)
    ax.set_xticks(x + width, ecg_pheno, fontsize=9)
    ax.legend(loc='upper left', ncols=4)
    plt.savefig(os.path.join(args.plot_path,"missing_mri2ecg.pdf"))
    fig.clf()

    multiplier = 0
    x = np.arange(len(mri_pheno))
    fig, ax = plt.subplots(layout='constrained', figsize=(len(x)*2, 4))
    scores = {'Our ECG':[np.mean(f1[pheno][1]) for pheno in mri_pheno],
            'Unimodal ECG':[np.mean(b1[pheno][1]) for pheno in mri_pheno],
            'DropFuse ECG':[np.mean(df1[pheno][1]) for pheno in mri_pheno],\
            }
    scores_error = {'Our ECG':[np.std(f1[pheno][1]) for pheno in mri_pheno],
                    'Unimodal ECG':[np.std(b1[pheno][1]) for pheno in mri_pheno],
                    'DropFuse ECG':[np.std(df1[pheno][1]) for pheno in mri_pheno],
                    }
    for attribute, measurement in scores.items():
        offset = width * multiplier
        rounded_list = [round(m*100)/100 for m in measurement]
        rects = ax.bar(x + offset, rounded_list, width, label=attribute)
        ax.errorbar(x + offset, rounded_list, yerr=scores_error[attribute], fmt="o", color="black")
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('R-squared')
    ax.set_title('Prediction of MRI derived phenotypes from ECG', fontsize=14)
    ax.set_xticks(x + width, mri_pheno)
    ax.legend(loc='upper left', ncols=4)
    plt.savefig(os.path.join(args.plot_path,"missing_ecg2mri.pdf"))
    fig.clf()

    multiplier = 0
    x = np.arange(len(diagnostics))
    fig, ax = plt.subplots(layout='constrained', figsize=(len(x)*2, 4))
    scores = {'Our ECG':[np.mean(f1[pheno][1]) for pheno in diagnostics],
            'Unimodal ECG':[np.mean(b1[pheno][1]) for pheno in diagnostics],
            'DropFuse ECG':[np.mean(df1[pheno][1]) for pheno in diagnostics],\
            }
    scores_error = {'Our ECG':[np.std(f1[pheno][1]) for pheno in diagnostics],
                    'Unimodal ECG':[np.std(b1[pheno][1]) for pheno in diagnostics],
                    'DropFuse ECG':[np.std(df1[pheno][1]) for pheno in diagnostics],
                    }
    for attribute, measurement in scores.items():
        offset = width * multiplier
        rounded_list = [round(m*100)/100 for m in measurement]
        rects = ax.bar(x + offset, rounded_list, width, label=attribute)
        ax.errorbar(x + offset, rounded_list, yerr=scores_error[attribute], fmt="o", color="black")
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('R-squared')
    ax.set_title('Prediction of diagnostics phenotypes from ECG', fontsize=14)
    ax.set_xticks(x + width, diagnostics_ticks)
    ax.legend(loc='upper left', ncols=4)
    plt.savefig(os.path.join(args.plot_path,"missing_ecg2diag.pdf"))
    fig.clf()

    multiplier = 0
    x = np.arange(len(diagnostics))
    fig, ax = plt.subplots(layout='constrained', figsize=(len(x)*2, 4))
    scores = {'Our MRI':[np.mean(f2[pheno][1]) for pheno in diagnostics],
            'Unimodal MRI':[np.mean(b2[pheno][1]) for pheno in diagnostics],
            'DropFuse MRI':[np.mean(df2[pheno][1]) for pheno in diagnostics],
            }
    scores_error = {'Our MRI':[np.std(f2[pheno][1]) for pheno in diagnostics],
                    'Unimodal MRI':[np.std(b2[pheno][1]) for pheno in diagnostics],
                    'DropFuse MRI':[np.std(df2[pheno][1]) for pheno in diagnostics],
                    }
    for attribute, measurement in scores.items():
        offset = width * multiplier
        rounded_list = [round(m*100)/100 for m in measurement]
        rects = ax.bar(x + offset, rounded_list, width, label=attribute)
        ax.errorbar(x + offset, rounded_list, yerr=scores_error[attribute], fmt="o", color="black")
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('R-squared')
    ax.set_title('Prediction of diagnostics from MRI', fontsize=14)
    ax.set_xticks(x + width, diagnostics_ticks)
    ax.legend(loc='upper left', ncols=4)
    plt.savefig(os.path.join(args.plot_path,"missing_mri2diag.pdf"))
    fig.clf()   



    # Different componenet analysis
    labels = ecg_pheno+mri_pheno
    labels_xticks = ecg_pheno+mri_pheno
    x = np.arange(len(labels))
    width = 0.2  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained', figsize=(1.6*len(labels), 4))
    scores = {'ECG-specific':[np.mean(s3[pheno][1]) for pheno in labels],
              'MRI-specific':[np.mean(s4[pheno][1]) for pheno in labels],
              'shared':[np.mean(s5[pheno][1]) for pheno in labels],
            }
    scores_error = {'ECG-specific':[np.std(s3[pheno][1]) for pheno in labels],
              'MRI-specific':[np.std(s4[pheno][1]) for pheno in labels],
              'shared':[np.std(s5[pheno][1]) for pheno in labels],
            }
    for attribute, measurement in scores.items():
        offset = width * multiplier
        rounded_list = [round(m*100)/100 for m in measurement]
        rects = ax.bar(x + offset, rounded_list, width, label=attribute)
        ax.errorbar(x + offset, rounded_list, yerr=scores_error[attribute], fmt="o", color="black")
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('R-squared')
    ax.set_title('Comparison of the predictive performance of representations', fontsize=14)
    ax.set_xticks(x + width, labels_xticks)#, rotation=40)
    ax.legend(loc='upper left', ncols=4)
    plt.savefig(os.path.join(args.plot_path,"component_prediction.pdf"))
    fig.clf()

    labels = diagnostics
    x = np.arange(len(labels))
    width = 0.2  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained', figsize=(1.6*len(labels), 4))
    scores = {'ECG-specific':[np.mean(s3[pheno][1]) for pheno in labels],
              'MRI-specific':[np.mean(s4[pheno][1]) for pheno in labels],
              'shared':[np.mean(s5[pheno][1]) for pheno in labels],
            }
    scores_error = {'ECG-specific':[np.std(s3[pheno][1]) for pheno in labels],
              'MRI-specific':[np.std(s4[pheno][1]) for pheno in labels],
              'shared':[np.std(s5[pheno][1]) for pheno in labels],
            }
    for attribute, measurement in scores.items():
        offset = width * multiplier
        rounded_list = [round(m*100)/100 for m in measurement]
        rects = ax.bar(x + offset, rounded_list, width, label=attribute)
        ax.errorbar(x + offset, rounded_list, yerr=scores_error[attribute], fmt="o", color="black")
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('AUROC')
    ax.set_title('Representation components performance for predicting diagnostic labels', fontsize=14)
    ax.set_xticks(x + width, diagnostics_ticks)#, rotation=40)
    ax.legend(loc='upper left', ncols=4)
    plt.savefig(os.path.join(args.plot_path,"diagnosis_component_prediction.pdf"))
    fig.clf()

    for phenotype in ecg_pheno+mri_pheno+shared_pheno+diagnostics:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        zm_mri = np.vstack(valid_pheno_df['zm_mri'])
        mask_mri = rep_disentangler.modality_masks['input_lax_4ch_heart_center_continuous'].binary_mask
        zm_mri = np.take(zm_mri, np.argwhere(mask_mri==1)[:,0], axis=1)
        # zm_mri = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(zm_mri)
        if zm_mri.shape[-1]!=0:
            zm_mri = PCA(n_components=2).fit_transform(zm_mri)
            axs[0].scatter(zm_mri[:,0], zm_mri[:,1], c=valid_pheno_df[phenotype].to_list(),
                            cmap=sns.cubehelix_palette(as_cmap=True))
            axs[0].set_title("Modality-specific (MRI)", fontsize=14)
        zs = np.vstack(valid_pheno_df['z_s'])
        # mask_shared = rep_disentangler.shared_mask.binary_mask
        # zs = np.take(zs, np.argwhere(mask_shared==1)[:,0], axis=1)
        # zs_mri = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(zs_mri)
        zs = PCA(n_components=2).fit_transform(zs)
        axs[1].scatter(zs[:,0], zs[:,1], c=valid_pheno_df[phenotype].to_list(),
                         cmap=sns.cubehelix_palette(as_cmap=True))
        axs[1].set_title("Shared", fontsize=14)
        zm_ecg = np.vstack(valid_pheno_df['zm_ecg'])
        mask_ecg = rep_disentangler.modality_masks['input_ecg_rest_median_raw_10_continuous'].binary_mask
        zm_ecg = np.take(zm_ecg, np.argwhere(mask_ecg==1)[:,0], axis=1)
        # zm_ecg = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(zm_ecg)
        if zm_ecg.shape[-1]!=0:
            zm_ecg = PCA(n_components=2).fit_transform(zm_ecg)
            axs[2].scatter(zm_ecg[:,0], zm_ecg[:,1], c=valid_pheno_df[phenotype].to_list(),
                            cmap=sns.cubehelix_palette(as_cmap=True))
            axs[2].set_title("Modality-specific (ECG)", fontsize=14)
        plt.savefig(os.path.join(args.plot_path,"scatter_%s.pdf"%phenotype))
        fig.clf()

    sys.stdout = orig_stdout
    f.close()


if __name__=="__main__":
    main()
