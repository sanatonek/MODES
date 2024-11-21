
import sys
import os
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

def main():
    orig_stdout = sys.stdout
    f = open('/home/sana/multimodal/logs/eval_ablation.txt', 'w')
    sys.stdout = f

    data_path = "/mnt/disks/google-annotated-cardiac-tensors-45k-2021-03-25/2020-09-21"
    # ecg_decoder_path = '/home/sana/ml4h/model_zoo/dropfuse/decoder_ecg_rest_median_raw_10.h5'
    # mri_decoder_path = '/home/sana/ml4h/model_zoo/dropfuse/decoder_lax_4ch_heart_center.h5'
    ecg_decoder_path = '/home/sana/model_ckpts/decoder_ecg_rest_median_raw_10.h5'
    ecg_encoder_path = '/home/sana/model_ckpts/encoder_ecg_rest_median_raw_10.h5'
    mri_encoder_path = '/home/sana/model_ckpts/encoder_lax_4ch_heart_center.h5'
    mri_decoder_path = '/home/sana/model_ckpts/decoder_lax_4ch_heart_center.h5'
    modality_names = ['input_ecg_rest_median_raw_10_continuous', 'input_lax_4ch_heart_center_continuous']
    data_paths = {'input_lax_4ch_heart_center_continuous':data_path, 'input_ecg_rest_median_raw_10_continuous':data_path}
    ckpt_nomask_path = "/home/sana/multimodal/ckpts_no_mask"
    ckpt_path = "/home/sana/multimodal/ckpts"
    plot_path = "/home/sana/multimodal/plots"

    z_s_size = 512#256
    z_m_size = 256#
    
    
    # Load pretrain models
    ecg_decoder, ecg_encoder, mri_encoder, mri_decoder = load_pretrained_models(ecg_decoder_path, ecg_encoder_path, mri_encoder_path, mri_decoder_path,
                                                                                shared_s=z_s_size, modality_specific_s=z_m_size)
    
    # Load data
    sample_list = get_id_list(data_path, from_file=True, file_name="/home/sana/multimodal/data_list.pkl")
    print("Total number of samples: ", len(sample_list))
    _, valid_loader, test_loader, list_ids, trainset = load_data(sample_list, data_paths, n_train=1000, train_ratio=0.9, test_ratio=0.05, get_trainset=True)

    rep_disentangler = MultimodalRep(encoders={'input_ecg_rest_median_raw_10_continuous': keras.models.clone_model(ecg_encoder), 'input_lax_4ch_heart_center_continuous':keras.models.clone_model(mri_encoder)}, 
                          decoders={'input_ecg_rest_median_raw_10_continuous': ecg_decoder, 'input_lax_4ch_heart_center_continuous':mri_decoder}, 
                          train_ids=list_ids[0], shared_size=z_s_size, modality_names=modality_names, 
                          z_sizes={'input_ecg_rest_median_raw_10_continuous':z_m_size, 'input_lax_4ch_heart_center_continuous':z_m_size},
                          modality_shapes={'input_ecg_rest_median_raw_10_continuous':(600, 12), 'input_lax_4ch_heart_center_continuous':(96, 96, 50)},
                          mask=True, ckpt_path=ckpt_path
                          )  
    rep_disentangler.load_from_checkpoint()
    rep_disentangler._set_trainable_mask(trainable=False)

    rep_disentangler_no_mask = MultimodalRep(encoders={'input_ecg_rest_median_raw_10_continuous': keras.models.clone_model(ecg_encoder), 'input_lax_4ch_heart_center_continuous':keras.models.clone_model(mri_encoder)}, 
                          decoders={'input_ecg_rest_median_raw_10_continuous': ecg_decoder, 'input_lax_4ch_heart_center_continuous':mri_decoder}, 
                          train_ids=list_ids[0], shared_size=356, modality_names=modality_names, 
                          z_sizes={'input_ecg_rest_median_raw_10_continuous':128, 'input_lax_4ch_heart_center_continuous':128},
                          modality_shapes={'input_ecg_rest_median_raw_10_continuous':(600, 12), 'input_lax_4ch_heart_center_continuous':(96, 96, 50)},
                          mask=False, ckpt_path=ckpt_nomask_path
                          )  
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
    # mri_pheno = ['LA_2Ch_vol_max', 'LA_2Ch_vol_min', 'LA_4Ch_vol_max', 'LA_4Ch_vol_min', 'LVEDV', 'LVEF',
    #              'LVESV', 'LVM', 'LVSV', 'RVEDV', 'RVEF', 'RVESV', 'RVSV']
    mri_pheno = ['LVEDV', 'LVESV', 'LVM', 'LVSV', 'RVEDV', 'RVESV', 'RVSV']
    shared_pheno = ['21003_Age-when-attended-assessment-centre_2_0', '21001_Body-mass-index-BMI_2_0']
    diagnostics = ['atrial_fibrillation_or_flutter',  'coronary_artery_disease']#'Sex_Male_0_0',
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


    # TODO: put this into a function
    print("\nEvaluating modality-specific ECG")
    s1 = phenotype_predictor(np.vstack(valid_pheno_df['zm_ecg']), valid_pheno_df, 
                             np.vstack(test_pheno_df['zm_ecg']), test_pheno_df,
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, 
                             mask=rep_disentangler.modality_masks['input_ecg_rest_median_raw_10_continuous'].binary_mask)
    s1_nm = phenotype_predictor(np.vstack(valid_pheno_df['zm_ecg (no mask)']), valid_pheno_df, 
                             np.vstack(test_pheno_df['zm_ecg (no mask)']), test_pheno_df,
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    print("\nEvaluating modality-specific MRI")
    s2 = phenotype_predictor(np.vstack(valid_pheno_df['zm_mri']), valid_pheno_df, 
                             np.vstack(test_pheno_df['zm_mri']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, 
                             mask=rep_disentangler.modality_masks['input_lax_4ch_heart_center_continuous'].binary_mask)
    s2_nm = phenotype_predictor(np.vstack(valid_pheno_df['zm_mri (no mask)']), valid_pheno_df, 
                             np.vstack(test_pheno_df['zm_mri (no mask)']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    print("\nEvaluating shared merged")
    s3 = phenotype_predictor(np.vstack(valid_pheno_df['z_s']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_s']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    s3_nm = phenotype_predictor(np.vstack(valid_pheno_df['z_s (no mask)']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_s (no mask)']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    print("\nEvaluating merged representations")
    m1 = phenotype_predictor(np.vstack(valid_pheno_df['z_merged']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_merged']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    m1_nm = phenotype_predictor(np.vstack(valid_pheno_df['z_merged (no mask)']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_merged (no mask)']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)


    plot_pheno_prediction_performance(phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, 
                                      perf_results=[s1, s2, s3, m1, s1_nm, s2_nm, s3_nm, m1_nm], 
                                      labels=['ECG', 'MRI', 'Shared', 'merged', 'ECG (no mask)', 'MRI (no mask)', 'Shared (no mask)', 'merged (no mask)'],
                                      tag='mask_all')

    labels = ecg_pheno+mri_pheno
    x = np.arange(len(labels))
    width = 0.2  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained', figsize=(len(x), 3))
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
    ax.set_title('Comparison of the predictive performance of representations with and without mask', fontsize=14)
    ax.set_xticks(x + width, labels)#, rotation=40)
    ax.legend(loc='upper left', ncols=4)
    plt.savefig(os.path.join(plot_path,"mask_no_mask_prediction.pdf"))
    fig.clf()

    # # Missing modality experiment
    # # labels = ecg_pheno+mri_pheno
    # # x = np.arange(len(labels))
    # width = 0.2  # the width of the bars
    # multiplier = 0
    # x = np.arange(len(ecg_pheno))
    # fig, ax = plt.subplots(layout='constrained', figsize=(len(x)*2, 4))
    # scores = {'Our MRI':[np.mean(f2[pheno][1]) for pheno in ecg_pheno],
    #         'Unimodal MRI':[np.mean(b2[pheno][1]) for pheno in ecg_pheno],
    #         'DropFuse MRI':[np.mean(df2[pheno][1]) for pheno in ecg_pheno],
    #         }
    # scores_error = {'Our MRI':[np.std(f2[pheno][1]) for pheno in ecg_pheno],
    #                 'Unimodal MRI':[np.std(b2[pheno][1]) for pheno in ecg_pheno],
    #                 'DropFuse MRI':[np.std(df2[pheno][1]) for pheno in ecg_pheno],
    #                 }
    # for attribute, measurement in scores.items():
    #     offset = width * multiplier
    #     rounded_list = [round(m*100)/100 for m in measurement]
    #     rects = ax.bar(x + offset, rounded_list, width, label=attribute)
    #     ax.errorbar(x + offset, rounded_list, yerr=scores_error[attribute], fmt="o", color="black")
    #     # ax.bar_label(rects, padding=3)
    #     multiplier += 1
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('R-squared')
    # ax.set_title('Prediction of ECG derived phenotypes from MRI', fontsize=14)
    # ax.set_xticks(x + width, ecg_pheno)#, rotation=40)
    # ax.legend(loc='upper left', ncols=4)
    # plt.savefig(os.path.join(plot_path,"missing_mri2ecg.pdf"))
    # fig.clf()

    # multiplier = 0
    # x = np.arange(len(mri_pheno))
    # fig, ax = plt.subplots(layout='constrained', figsize=(len(x)*2, 4))
    # scores = {'Our ECG':[np.mean(f1[pheno][1]) for pheno in mri_pheno],
    #         'Unimodal ECG':[np.mean(b1[pheno][1]) for pheno in mri_pheno],
    #         'DropFuse ECG':[np.mean(df1[pheno][1]) for pheno in mri_pheno],\
    #         }
    # scores_error = {'Our ECG':[np.std(f1[pheno][1]) for pheno in mri_pheno],
    #                 'Unimodal ECG':[np.std(b1[pheno][1]) for pheno in mri_pheno],
    #                 'DropFuse ECG':[np.std(df1[pheno][1]) for pheno in mri_pheno],
    #                 }
    # for attribute, measurement in scores.items():
    #     offset = width * multiplier
    #     rounded_list = [round(m*100)/100 for m in measurement]
    #     rects = ax.bar(x + offset, rounded_list, width, label=attribute)
    #     ax.errorbar(x + offset, rounded_list, yerr=scores_error[attribute], fmt="o", color="black")
    #     # ax.bar_label(rects, padding=3)
    #     multiplier += 1
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('R-squared')
    # ax.set_title('Prediction of MRI derived phenotypes from ECG', fontsize=14)
    # ax.set_xticks(x + width, mri_pheno)#, rotation=40)
    # ax.legend(loc='upper left', ncols=4)
    # plt.savefig(os.path.join(plot_path,"missing_ecg2mri.pdf"))
    # fig.clf()



    # # Different componenet analysis
    # labels = ecg_pheno+mri_pheno
    # x = np.arange(len(labels))
    # width = 0.2  # the width of the bars
    # multiplier = 0
    # fig, ax = plt.subplots(layout='constrained', figsize=(15, 4))
    # scores = {'ECG-specific':[np.mean(s3[pheno][1]) for pheno in labels],
    #           'MRI-specific':[np.mean(s4[pheno][1]) for pheno in labels],
    #           'shared':[np.mean(s5[pheno][1]) for pheno in labels],
    #           'Unimodal ECG':[np.mean(b1[pheno][1]) for pheno in labels],
    #           'Unimodal MRI':[np.mean(b2[pheno][1]) for pheno in labels],
    #         }
    # scores_error = {'ECG-specific':[np.std(s3[pheno][1]) for pheno in labels],
    #           'MRI-specific':[np.std(s4[pheno][1]) for pheno in labels],
    #           'shared':[np.std(s5[pheno][1]) for pheno in labels],
    #           'Unimodal ECG':[np.std(b1[pheno][1]) for pheno in labels],
    #           'Unimodal MRI':[np.std(b2[pheno][1]) for pheno in labels],
    #         }
    # for attribute, measurement in scores.items():
    #     offset = width * multiplier
    #     rounded_list = [round(m*100)/100 for m in measurement]
    #     rects = ax.bar(x + offset, rounded_list, width, label=attribute)
    #     ax.errorbar(x + offset, rounded_list, yerr=scores_error[attribute], fmt="o", color="black")
    #     # ax.bar_label(rects, padding=3)
    #     multiplier += 1
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('R-squared')
    # ax.set_title('Comparison of the predictive performance of representations', fontsize=14)
    # ax.set_xticks(x + width, labels)#, rotation=40)
    # ax.legend(loc='upper left', ncols=4)
    # plt.savefig(os.path.join(plot_path,"component_prediction.pdf"))
    # fig.clf()

    sys.stdout = orig_stdout
    f.close()


if __name__=="__main__":
    # with open('/home/sana/multimodal/logs/out.txt', 'w') as sys.stdout:
    main()
