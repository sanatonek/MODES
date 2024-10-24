
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append('/home/sana/ml4h')
sys.path.append('/home/sana')
import tensorflow as tf

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
    f = open('/home/sana/multimodal/logs/eval_out.txt', 'w')
    sys.stdout = f

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

    z_s_size = 356#256
    z_m_size = 128#
    
    
    # Load pretrain models
    ecg_decoder, ecg_encoder, mri_encoder, mri_decoder = load_pretrained_models(ecg_decoder_path, ecg_encoder_path, mri_encoder_path, mri_decoder_path,
                                                                                shared_s=z_s_size, modality_specific_s=z_m_size)
    # print(list(pd.read_csv("/home/sana/tensors_all_union.csv").columns))
    
    # Load data
    sample_list = get_id_list(data_path, from_file=True, file_name="/home/sana/multimodal/data_list.pkl")
    print("Total number of samples: ", len(sample_list))
    _, valid_loader, test_loader, list_ids, trainset = load_data(sample_list, data_path, train_ratio=0.9, test_ratio=0.05, get_trainset=True)

    rep_disentangler = MultimodalRep(encoders={'input_ecg_rest_median_raw_10_continuous': keras.models.clone_model(ecg_encoder), 'input_lax_4ch_heart_center_continuous':keras.models.clone_model(mri_encoder)}, 
                          decoders={'input_ecg_rest_median_raw_10_continuous': ecg_decoder, 'input_lax_4ch_heart_center_continuous':mri_decoder}, 
                          train_ids=list_ids[0], shared_size=z_s_size, modality_names=modality_names, 
                          z_sizes={'input_ecg_rest_median_raw_10_continuous':z_m_size, 'input_lax_4ch_heart_center_continuous':z_m_size},
                          modality_shapes={'input_ecg_rest_median_raw_10_continuous':(600, 12), 'input_lax_4ch_heart_center_continuous':(96, 96, 50)},
                          mask=False
                          )  
    rep_disentangler.load_from_checkpoint("/home/sana/multimodal/ckpts")
    rep_disentangler._set_trainable_mask(trainable=False)
    # sys.exit("Error message")
    # Plot reconstructed samples
    test_batch, test_ids = next(iter(test_loader))
    valid_batch, valid_ids = next(iter(valid_loader))
    test_ids = test_ids.tolist()
    valid_ids = valid_ids.tolist()
    z_s_test, z_m_test = rep_disentangler.encode(test_batch)
    z_merged_test = rep_disentangler.merge_representations(test_batch)
    z_s_valid, z_m_valid = rep_disentangler.encode(valid_batch)
    z_merged_valid = rep_disentangler.merge_representations(valid_batch)
    baseline_z_test = {'input_ecg_rest_median_raw_10_continuous':ecg_encoder(tf.convert_to_tensor(test_batch['input_ecg_rest_median_raw_10_continuous'])), 
                       'input_lax_4ch_heart_center_continuous':mri_encoder(tf.convert_to_tensor(test_batch['input_lax_4ch_heart_center_continuous']))}
    baseline_z_valid = {'input_ecg_rest_median_raw_10_continuous':ecg_encoder(tf.convert_to_tensor(valid_batch['input_ecg_rest_median_raw_10_continuous'])), 
                       'input_lax_4ch_heart_center_continuous':mri_encoder(tf.convert_to_tensor(valid_batch['input_lax_4ch_heart_center_continuous']))}
    i = 0
    for data_batch, test_id_batch in test_loader:
        z_s_all, z_m_all = rep_disentangler.encode(data_batch)
        z_merged_test = tf.concat([z_merged_test, rep_disentangler.merge_representations(data_batch)], 0)
        for key in z_s_all.keys():
            z_s_test[key] = tf.concat([z_s_test[key], z_s_all[key]], 0)
            z_m_test[key] = tf.concat([z_m_test[key], z_m_all[key]], 0)
            if 'lax' in key:
                baseline_z_test[key] = tf.concat([baseline_z_test[key], mri_encoder(tf.convert_to_tensor(data_batch[key]))], 0)
            if 'ecg' in key:
                baseline_z_test[key] = tf.concat([baseline_z_test[key], ecg_encoder(tf.convert_to_tensor(data_batch[key]))], 0)
            # test_batch[key] = tf.concat([test_batch[key], data_batch[key]], 0)
        test_ids.extend(test_id_batch.tolist())
        i += 1
        if i>=1:
            break
    print('**********', z_merged_test.shape)

    i = 0
    for data_batch, valid_id_batch in valid_loader:
        z_s_all, z_m_all = rep_disentangler.encode(data_batch)
        z_merged_valid = tf.concat([z_merged_valid, rep_disentangler.merge_representations(data_batch)], 0)
        for key in z_s_all.keys():
            z_s_valid[key] = tf.concat([z_s_valid[key], z_s_all[key]], 0)
            z_m_valid[key] = tf.concat([z_m_valid[key], z_m_all[key]], 0)
            if 'lax' in key:
                baseline_z_valid[key] = tf.concat([baseline_z_valid[key], mri_encoder(tf.convert_to_tensor(data_batch[key]))], 0)
            if 'ecg' in key:
                baseline_z_valid[key] = tf.concat([baseline_z_valid[key], ecg_encoder(tf.convert_to_tensor(data_batch[key]))], 0)
            # valid_batch[key] = tf.concat([valid_batch[key], data_batch[key]], 0)
        valid_ids.extend(valid_id_batch.tolist()) 
        # i += 1
        # if i>=50:
        #     break 
    print('**********', z_merged_valid.shape)

    
    # plot some generated samples
    rnd_id = np.random.randint(0,len(test_batch))
    generated_samples, generated_ecgs = [], []
    mri_pcas, low_mri_pcas, ecg_pcas, low_ecg_pcas = [], [], [], []
    for i in range(3):
        generated_mri, max_pc_mri_inds, min_pc_mri_inds = (rep_disentangler.generate_sample(test_batch, rnd_id, n_pca=i, variation=True,
                                                            ref_modality='input_ecg_rest_median_raw_10_continuous', 
                                                            target_modality='input_lax_4ch_heart_center_continuous', n_samples=3))
        generated_ecg, max_pc_ecg_inds, min_pc_ecg_inds = (rep_disentangler.generate_sample(test_batch, rnd_id, n_pca=i, variation=True,
                                                            ref_modality='input_lax_4ch_heart_center_continuous', 
                                                            target_modality='input_ecg_rest_median_raw_10_continuous', n_samples=3))
        generated_samples.append(generated_mri)
        generated_ecgs.append(generated_ecg)
        mri_pcas.extend([trainset[ind][0]['input_lax_4ch_heart_center_continuous'] for ind in (min_pc_mri_inds+max_pc_mri_inds)])
        # low_mri_pcas.append([trainset[ind]['input_lax_4ch_heart_center_continuous'] for ind in min_pc_mri_inds])
        ecg_pcas.extend([trainset[ind][0]['input_ecg_rest_median_raw_10_continuous'] for ind in (min_pc_ecg_inds+max_pc_ecg_inds)])
        # low_ecg_pcas.append([trainset[ind]['input_ecg_rest_median_raw_10_continuous'] for ind in min_pc_ecg_inds])
        
    plot_sample(tf.concat(generated_samples,0), num_cols=len(generated_samples[0]), num_rows=3, 
                save_path="/home/sana/multimodal/plots/generated_distribution_mri.pdf")
    plot_sample(tf.concat(generated_ecgs,0), num_cols=len(generated_samples[0]), num_rows=3, 
                save_path="/home/sana/multimodal/plots/generated_distribution_ecg.pdf")
    plot_sample(mri_pcas, num_cols=len(generated_samples[0]), num_rows=3, 
                save_path="/home/sana/multimodal/plots/pca_mri.pdf")
    plot_sample(ecg_pcas, num_cols=len(generated_samples[0]), num_rows=3, 
                save_path="/home/sana/multimodal/plots/pca_ecg.pdf")

    
    generated_mod, probs, ref_sample = rep_disentangler.generate_sample(test_batch, rnd_id, variation=False,
                                    ref_modality='input_ecg_rest_median_raw_10_continuous', 
                                    target_modality='input_lax_4ch_heart_center_continuous', n_samples=8)
    plot_sample([ref_sample]+[g for g in generated_mod], num_cols=8+1, num_rows=1, 
                save_path="/home/sana/multimodal/plots/generated_similar_mri.pdf")
    plt.bar(np.arange(8), probs)
    plt.title("Probability of generated samples")
    plt.savefig("/home/sana/multimodal/plots/generated_similar_mri_probs.pdf")
    
    generated_mod, probs, ref_sample = rep_disentangler.generate_sample(test_batch, rnd_id, variation=False,
                                    ref_modality='input_lax_4ch_heart_center_continuous', 
                                    target_modality='input_ecg_rest_median_raw_10_continuous', n_samples=8)
    plot_sample([ref_sample]+[g for g in generated_mod], num_cols=8+1, num_rows=1, 
                save_path="/home/sana/multimodal/plots/generated_similar_ecg.pdf")



    reconstructed_samples = rep_disentangler.decode(z_s_test, z_m_test)
    z_s_mixed = {}
    z_s_mixed[modality_names[0]] = z_s_test[modality_names[1]]
    z_s_mixed[modality_names[1]] = z_s_test[modality_names[0]]
    reconstructed_mixed_samples = rep_disentangler.decode(z_s_mixed, z_m_test)
    for m_name in modality_names:
        plot_sample(test_batch[m_name], num_cols=4, num_rows=1,
                    save_path="/home/sana/multimodal/plots/original_%s.pdf"%m_name)
        plot_sample(reconstructed_samples[m_name], num_cols=4, num_rows=1,
                    save_path="/home/sana/multimodal/plots/reconstructed_%s.pdf"%m_name)
        plot_sample(reconstructed_mixed_samples[m_name], num_cols=4, num_rows=1,
                    save_path="/home/sana/multimodal/plots/reconstructed_mixed_%s.pdf"%m_name)

        # Plot top pca samples
        # pos_samples_s, neg_samples_s, pos_samples_m, neg_samples_m = pca_analysis(test_batch, rep_disentangler, modality_name=m_name, percentile=0.1)
        # plot_sample(pos_samples_s, num_cols=4, num_rows=1,
        #             save_path="/home/sana/multimodal/plots/top_pca_shared_%s.pdf"%m_name)
        # plot_sample(neg_samples_s, num_cols=4, num_rows=1,
        #             save_path="/home/sana/multimodal/plots/bottom_pca_shared_%s.pdf"%m_name)
        # plot_sample(pos_samples_m, num_cols=4, num_rows=1,
        #             save_path="/home/sana/multimodal/plots/top_pca_mod_%s.pdf"%m_name)
        # plot_sample(neg_samples_m, num_cols=4, num_rows=1,
        #             save_path="/home/sana/multimodal/plots/bottom_pca_mod_%s.pdf"%m_name)
        
        # plot correlations
        corr_mat = np.corrcoef(np.concatenate([z_m_test[m_name], z_s_test[m_name]], axis=-1).T)
        plt.matshow(corr_mat)
        plt.savefig("/home/sana/multimodal/plots/%s_correlations.pdf"%m_name)
        plt.close()
        
    ## Modality specific phenotypes
    ecg_pheno = ['PQInterval', 'QTInterval','QTCInterval','QRSDuration','RRInterval']
    # mri_pheno = ['LA_2Ch_vol_max', 'LA_2Ch_vol_min', 'LA_4Ch_vol_max', 'LA_4Ch_vol_min', 'LVEDV', 'LVEF',
    #              'LVESV', 'LVM', 'LVSV', 'RVEDV', 'RVEF', 'RVESV', 'RVSV']
    mri_pheno = ['LA_2Ch_vol_max', 'LA_2Ch_vol_min', 'LVEDV',
                 'LVESV', 'LVM', 'LVSV', 'RVEDV', 'RVESV', 'RVSV']
    shared_pheno = ['21003_Age-when-attended-assessment-centre_2_0', '21001_Body-mass-index-BMI_2_0']
    diagnostics = ['atrial_fibrillation_or_flutter', 'Sex_Male_0_0']
    phenotype_df = pd.read_csv("/home/sana/tensors_all_union.csv")[ecg_pheno+mri_pheno+shared_pheno+diagnostics+['fpath']]
    phenotype_df['fpath'] = phenotype_df['fpath'].astype(int)
    phenotype_df.dropna(inplace=True)
    zs_size = int(np.sum(rep_disentangler.shared_mask.binary_mask))
    rep_df = pd.DataFrame({'fpath': test_ids, 'zs_ecg':z_s_test['input_ecg_rest_median_raw_10_continuous'].numpy().tolist(),
                           'zs_mri':z_s_test['input_lax_4ch_heart_center_continuous'].numpy().tolist() ,
                           'zm_ecg':z_m_test['input_ecg_rest_median_raw_10_continuous'].numpy().tolist() ,
                           'zm_mri':z_m_test['input_lax_4ch_heart_center_continuous'].numpy().tolist(),
                           'z_s': z_merged_test[:, -zs_size:].numpy().tolist(),
                           'z_baseline_ecg': baseline_z_test['input_ecg_rest_median_raw_10_continuous'].numpy().tolist(),
                           'z_baseline_mri': baseline_z_test['input_lax_4ch_heart_center_continuous'].numpy().tolist(),
                           'z_merged': z_merged_test.numpy().tolist()}) 
    rep_df_valid = pd.DataFrame({'fpath': valid_ids, 'zs_ecg':z_s_valid['input_ecg_rest_median_raw_10_continuous'].numpy().tolist(),
                           'zs_mri':z_s_valid['input_lax_4ch_heart_center_continuous'].numpy().tolist() ,
                           'zm_ecg':z_m_valid['input_ecg_rest_median_raw_10_continuous'].numpy().tolist() ,
                           'zm_mri':z_m_valid['input_lax_4ch_heart_center_continuous'].numpy().tolist(),
                           'z_s': z_merged_valid[:, -zs_size:].numpy().tolist(),
                            'z_baseline_ecg': baseline_z_valid['input_ecg_rest_median_raw_10_continuous'].numpy().tolist(),
                            'z_baseline_mri': baseline_z_valid['input_lax_4ch_heart_center_continuous'].numpy().tolist(),
                            'z_merged': z_merged_valid.numpy().tolist()})
    valid_pheno_df = pd.merge(phenotype_df, rep_df_valid, on='fpath')
    pheno_mean = valid_pheno_df[ecg_pheno+mri_pheno+shared_pheno].mean()
    pheno_std = valid_pheno_df[ecg_pheno+mri_pheno+shared_pheno].std()
    valid_pheno_df[ecg_pheno+mri_pheno+shared_pheno] = (valid_pheno_df[ecg_pheno+mri_pheno+shared_pheno]-pheno_mean)/pheno_std
    test_pheno_df = pd.merge(phenotype_df, rep_df, on='fpath')
    test_pheno_df[ecg_pheno+mri_pheno+shared_pheno] = (test_pheno_df[ecg_pheno+mri_pheno+shared_pheno]-pheno_mean)/pheno_std

    ecg_ecg_score, ecg_mri_score = cluster_test(rep_df, n_clusters=5, original_modality="ecg")
    mri_mri_score, mri_ecg_score = cluster_test(rep_df, n_clusters=5, original_modality="mri")
    print("ECG representation (m+s) matching with baseline ECG: ", ecg_ecg_score)
    print("modality-specific MRI representation matching with baseline ECG: ", ecg_mri_score)
    print("MRI representation (m+s) matching with baseline MRI: ", mri_mri_score)
    print("modality-specific ECG representation matching with baseline MRI: ", mri_ecg_score)

    # TODO: put this into a function
    print("\nEvaluating shared ECG")
    s1 = phenotype_predictor(np.vstack(valid_pheno_df['zs_ecg']), valid_pheno_df, 
                             np.vstack(test_pheno_df['zs_ecg']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, 
                             mask=rep_disentangler.shared_mask.binary_mask)
    print("\nEvaluating shared MRI")
    s2 = phenotype_predictor(np.vstack(valid_pheno_df['zs_mri']), valid_pheno_df, 
                             np.vstack(test_pheno_df['zs_mri']), test_pheno_df,
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, 
                             mask=rep_disentangler.shared_mask.binary_mask)
    print("\nEvaluating modality-specific ECG")
    s3 = phenotype_predictor(np.vstack(valid_pheno_df['zm_ecg']), valid_pheno_df, 
                             np.vstack(test_pheno_df['zm_ecg']), test_pheno_df,
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, 
                             mask=rep_disentangler.modality_masks['input_ecg_rest_median_raw_10_continuous'].binary_mask)
    print("\nEvaluating modality-specific MRI")
    s4 = phenotype_predictor(np.vstack(valid_pheno_df['zm_mri']), valid_pheno_df, 
                             np.vstack(test_pheno_df['zm_mri']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, 
                             mask=rep_disentangler.modality_masks['input_lax_4ch_heart_center_continuous'].binary_mask)
    print("\nEvaluating shared merged")
    s5 = phenotype_predictor(np.vstack(valid_pheno_df['z_s']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_s']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    print("\nEvaluating full ECG")
    f1 = phenotype_predictor(np.concatenate([np.vstack(valid_pheno_df['zm_ecg']),np.vstack(valid_pheno_df['zs_ecg'])],-1),
                             valid_pheno_df,
                             np.concatenate([np.vstack(test_pheno_df['zm_ecg']),np.vstack(test_pheno_df['zs_ecg'])],-1), 
                             test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, 
                             mask=np.concatenate([rep_disentangler.modality_masks['input_ecg_rest_median_raw_10_continuous'].binary_mask, rep_disentangler.shared_mask.binary_mask], 0)) 
    print("\nEvaluating full MRI")
    f2 = phenotype_predictor(np.concatenate([np.vstack(valid_pheno_df['zm_mri']),np.vstack(valid_pheno_df['zs_mri'])],-1),
                             valid_pheno_df,
                             np.concatenate([np.vstack(test_pheno_df['zm_mri']),np.vstack(test_pheno_df['zs_mri'])],-1), 
                             test_pheno_df,phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, 
                             mask=np.concatenate([rep_disentangler.modality_masks['input_lax_4ch_heart_center_continuous'].binary_mask, rep_disentangler.shared_mask.binary_mask], 0))
    print("\nEvaluating baseline ECG")
    b1 = phenotype_predictor(np.vstack(valid_pheno_df['z_baseline_ecg']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_baseline_ecg']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    print("\nEvaluating baseline MRI")
    b2 = phenotype_predictor(np.vstack(valid_pheno_df['z_baseline_mri']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_baseline_mri']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    print("\nEvaluating merged representations")
    m1 = phenotype_predictor(np.vstack(valid_pheno_df['z_merged']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_merged']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    print("\nEvaluating concatenated baselines")
    m2 = phenotype_predictor(np.concatenate([np.vstack(valid_pheno_df['z_baseline_mri']),np.vstack(valid_pheno_df['z_baseline_ecg'])],-1), 
                             valid_pheno_df, 
                             np.concatenate([np.vstack(test_pheno_df['z_baseline_mri']),np.vstack(test_pheno_df['z_baseline_ecg'])],-1), 
                             test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)

    print("\nEvaluating concatenated baselines (PCA)")
    m2_pca = phenotype_predictor(np.concatenate([np.vstack(valid_pheno_df['z_baseline_mri']),np.vstack(valid_pheno_df['z_baseline_ecg'])],-1), 
                             valid_pheno_df, 
                             np.concatenate([np.vstack(test_pheno_df['z_baseline_mri']),np.vstack(test_pheno_df['z_baseline_ecg'])],-1), 
                             test_pheno_df, n_pca=rep_disentangler.merged_size,
                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics) 

    # Generalizability test
    # gen_test_ours, gen_test_pca, gen_test_concat = [], [], []
    # ratios = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    # for ratio in ratios:
    #     chopped_df = valid_pheno_df[:int(ratio*len(valid_pheno_df))]
    #     acc = phenotype_predictor(np.vstack(chopped_df['z_merged']), chopped_df, 
    #                             np.vstack(test_pheno_df['z_merged']), test_pheno_df, 
    #                             phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    #     acc_pca = phenotype_predictor(np.concatenate([np.vstack(chopped_df['z_baseline_mri']),np.vstack(chopped_df['z_baseline_ecg'])],-1),
    #                                 chopped_df, 
    #                                 np.concatenate([np.vstack(test_pheno_df['z_baseline_mri']),np.vstack(test_pheno_df['z_baseline_ecg'])],-1), 
    #                                 test_pheno_df, n_pca=rep_disentangler.merged_size,
    #                                 phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics) 
    #     acc_concat = phenotype_predictor(np.concatenate([np.vstack(chopped_df['z_baseline_mri']),np.vstack(chopped_df['z_baseline_ecg'])],-1),
    #                                 chopped_df, 
    #                                 np.concatenate([np.vstack(test_pheno_df['z_baseline_mri']),np.vstack(test_pheno_df['z_baseline_ecg'])],-1), 
    #                                 test_pheno_df, phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics)
    #     gen_test_ours.append(acc)
    #     gen_test_pca.append(acc_pca)
    #     gen_test_concat.append(acc_concat)
    
    plot_pheno_prediction_performance(phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, 
                                      perf_results=[b1, b2, m1, m2], 
                                      labels=['ecg rep.', 'mri rep.', 'merged rep.', 'concat. re.'],
                                      tag='overall_perf')
    plot_pheno_prediction_performance(phenotypes=ecg_pheno+mri_pheno+shared_pheno+diagnostics, 
                                      perf_results=[s3, s4, s5, m1, b1, b2, m2, m2_pca], 
                                      labels=['ECG', 'MRI', 'Shared', 'merged', 'ECG unimodal', 'MRI unimodal', 'concatenated', 'pca'],
                                      tag='all')

    labels = ecg_pheno+mri_pheno
    x = np.arange(len(labels))
    width = 0.2  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained', figsize=(20, 4))
    scores = {'Ours':[np.mean(m1[pheno][1]) for pheno in labels],
            'Concatenated':[np.mean(m2[pheno][1]) for pheno in labels],
            # 'PCA':[np.mean(m2_pca[pheno][1]) for pheno in labels],
            'Unimodal ECG':[np.mean(b1[pheno][1]) for pheno in labels],
            'Unimodal MRI':[np.mean(b2[pheno][1]) for pheno in labels],
            }
    scores_error = {'Ours':[np.std(m1[pheno][1]) for pheno in labels],
                    'Concatenated':[np.std(m2[pheno][1]) for pheno in labels],
                    # 'PCA':[np.std(m2_pca[pheno][1]) for pheno in labels],
                    'Unimodal ECG':[np.std(b1[pheno][1]) for pheno in labels],
                    'Unimodal MRI':[np.std(b2[pheno][1]) for pheno in labels],
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
    ax.set_title('Comparison of the predictive performance of representations')
    ax.set_xticks(x + width, labels, rotation=70)
    ax.legend(loc='upper left', ncols=4)
    plt.savefig("/home/sana/multimodal/plots/phenotype_prediction.pdf")
    fig.clf()

    # Missing modality experiment
    labels = ecg_pheno+mri_pheno
    x = np.arange(len(labels))
    width = 0.2  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained', figsize=(20, 4))
    scores = {'Our ECG':[np.mean(f1[pheno][1]) for pheno in labels],
            'Unimodal ECG':[np.mean(b1[pheno][1]) for pheno in labels],
            'Our MRI':[np.mean(f2[pheno][1]) for pheno in labels],
            'Unimodal MRI':[np.mean(b2[pheno][1]) for pheno in labels],
            }
    scores_error = {'Our ECG':[np.std(f1[pheno][1]) for pheno in labels],
                    'Unimodal ECG':[np.std(b1[pheno][1]) for pheno in labels],
                    'Our MRI':[np.std(f2[pheno][1]) for pheno in labels],
                    'Unimodal MRI':[np.std(b2[pheno][1]) for pheno in labels],
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
    ax.set_title('Comparison of the predictive performance of representations')
    ax.set_xticks(x + width, labels, rotation=70)
    ax.legend(loc='upper left', ncols=4)
    plt.savefig("/home/sana/multimodal/plots/missing_modality_prediction.pdf")
    fig.clf()

    # Different componenet analysis
    labels = ecg_pheno+mri_pheno
    x = np.arange(len(labels))
    width = 0.2  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained', figsize=(15, 4))
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
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('R-squared')
    ax.set_title('Comparison of the predictive performance of representations')
    ax.set_xticks(x + width, labels, rotation=70)
    ax.legend(loc='upper left', ncols=4)
    plt.savefig("/home/sana/multimodal/plots/component_prediction.pdf")
    fig.clf()

    for phenotype in ecg_pheno+mri_pheno+shared_pheno+diagnostics:
    #     labels = ['ecg_zs', 'ecg_zm', 'mri_zs', 'mri_zm', 'z_s', 'ecg rep.', 'mri rep.', 'merged rep.', 'concat. re.', 'concat. rep. pca']
    #     x = np.arange(len(labels))  # the label locations
    #     width = 0.4  # the width of the bars
    #     multiplier = 0
    #     fig, ax = plt.subplots(layout='constrained', figsize=(14, 4))
    #     scores = {'train':(s1[phenotype][0],s3[phenotype][0],s2[phenotype][0],s4[phenotype][0],s5[phenotype][0],b1[phenotype][0],b2[phenotype][1],m1[phenotype][0],m2[phenotype][0], m2_pca[phenotype][0]),
    #      'test':(s1[phenotype][1],s3[phenotype][1],s2[phenotype][1],s4[phenotype][1],s5[phenotype][1],b1[phenotype][1],b2[phenotype][1],m1[phenotype][1],m2[phenotype][1], m2_pca[phenotype][1])}
    #     for attribute, measurement in scores.items():
    #         offset = width * multiplier
    #         rounded_list = [round(m*100)/100 for m in measurement]
    #         rects = ax.bar(x + offset, rounded_list, width, label=attribute)
    #         ax.bar_label(rects, padding=3)
    #         multiplier += 1
    #     # Add some text for labels, title and custom x-axis tick labels, etc.
    #     ax.set_ylabel('R^2')
    #     ax.set_title(phenotype)
    #     ax.set_xticks(x + width, labels, rotation=70)
    #     # ax.xticks(rotation=70)
    #     ax.legend(loc='upper left', ncols=4)
    #     plt.savefig("/home/sana/multimodal/plots/%s.pdf"%phenotype)
    #     fig.clf()


        # Missing modality experiment
        # labels = ['Our ECG rep.', 'Baseline ECG rep.', 'Our MRI rep.', 'Baseline MRI rep.']
        # x = np.arange(len(labels))  # the label locations
        # width = 0.3  # the width of the bars
        # multiplier = 0
        # fig, ax = plt.subplots(layout='constrained', figsize=(8, 4))
        # scores = {'train':(f1[phenotype][0],b1[phenotype][0],f2[phenotype][0],b2[phenotype][0]),
        #  'test':(f1[phenotype][1],b1[phenotype][1],f2[phenotype][1],b2[phenotype][1])}
        # for attribute, measurement in scores.items():
        #     offset = width * multiplier
        #     rounded_list = [round(np.mean(m)*100)/100 for m in measurement]
        #     rects = ax.bar(x + offset, rounded_list, width, label=attribute)
        #     ax.bar_label(rects, padding=3)
        #     multiplier += 1
        # # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax.set_ylabel('R^2')
        # ax.set_title(phenotype)
        # ax.set_xticks(x + width, labels, rotation=70)
        # # ax.xticks(rotation=70)
        # ax.legend(loc='upper left', ncols=4)
        # ax.set_title(phenotype)
        # plt.savefig("/home/sana/multimodal/plots/missingness_%s.pdf"%phenotype)
        # fig.clf()

        
        # fig = plt.figure(figsize=(10, 5))
        # plt.plot(ratios, [g[phenotype][0] for g in gen_test_ours], label='Train merged')
        # plt.plot(ratios, [g[phenotype][1] for g in gen_test_ours], label='Test merged')
        # plt.plot(ratios, [g[phenotype][0] for g in gen_test_pca], label='Train pca')
        # plt.plot(ratios, [g[phenotype][1] for g in gen_test_pca], label='Test pca')
        # plt.plot(ratios, [g[phenotype][0] for g in gen_test_concat], label='Train concat.')
        # plt.plot(ratios, [g[phenotype][1] for g in gen_test_concat], label='Test concat.')
        # plt.legend()
        # plt.ylim([-0.5,1])
        # plt.savefig("/home/sana/multimodal/plots/generalizability_%s.pdf"%phenotype)
        # fig.clf()

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        zm_mri = np.vstack(valid_pheno_df['zm_mri'])
        mask_mri = rep_disentangler.modality_masks['input_lax_4ch_heart_center_continuous'].binary_mask
        zm_mri = np.take(zm_mri, np.argwhere(mask_mri==1)[:,0], axis=1)
        # zm_mri = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(zm_mri)
        if zm_mri.shape[-1]!=0:
            zm_mri = PCA(n_components=2).fit_transform(zm_mri)
            axs[0].scatter(zm_mri[:,0], zm_mri[:,1], c=valid_pheno_df[phenotype].to_list(),
                            cmap=sns.cubehelix_palette(as_cmap=True))
            axs[0].set_title("Modality-specific (MRI)")
        zs = np.vstack(valid_pheno_df['z_s'])
        # mask_shared = rep_disentangler.shared_mask.binary_mask
        # zs = np.take(zs, np.argwhere(mask_shared==1)[:,0], axis=1)
        # zs_mri = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(zs_mri)
        zs = PCA(n_components=2).fit_transform(zs)
        axs[1].scatter(zs[:,0], zs[:,1], c=valid_pheno_df[phenotype].to_list(),
                         cmap=sns.cubehelix_palette(as_cmap=True))
        axs[1].set_title("Shared")
        zm_ecg = np.vstack(valid_pheno_df['zm_ecg'])
        mask_ecg = rep_disentangler.modality_masks['input_ecg_rest_median_raw_10_continuous'].binary_mask
        zm_ecg = np.take(zm_ecg, np.argwhere(mask_ecg==1)[:,0], axis=1)
        # zm_ecg = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(zm_ecg)
        if zm_ecg.shape[-1]!=0:
            zm_ecg = PCA(n_components=2).fit_transform(zm_ecg)
            axs[2].scatter(zm_ecg[:,0], zm_ecg[:,1], c=valid_pheno_df[phenotype].to_list(),
                            cmap=sns.cubehelix_palette(as_cmap=True))
            axs[2].set_title("Modality-specific (ECG)")
        # zs_ecg = np.vstack(test_pheno_df['zs_ecg'])
        # zs_ecg = np.take(zs_ecg, np.argwhere(mask_shared==1)[:,0], axis=1)
        # # zs_ecg = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(zs_ecg)
        # zs_ecg = PCA(n_components=2).fit_transform(zs_ecg)
        # axs[1,1].scatter(zs_ecg[:,0], zs_ecg[:,1], c=test_pheno_df[phenotype].to_list(),
        #                  cmap=sns.cubehelix_palette(as_cmap=True))
        # axs[1,1].set_title("Shared (ECG)")
        plt.savefig("/home/sana/multimodal/plots/scatter_%s.pdf"%phenotype)
        fig.clf()

    sys.stdout = orig_stdout
    f.close()


if __name__=="__main__":
    # with open('/home/sana/multimodal/logs/out.txt', 'w') as sys.stdout:
    main()
