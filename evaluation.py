
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

from multimodal.contrastive import MultimodalRep
from multimodal.utils import load_data, get_id_list, plot_sample, pca_analysis, phenotype_predictor, load_pretrained_models, cluster_test

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

    z_s_size = 256
    z_m_size = 64
    
    # Load pretrain models
    ecg_decoder, ecg_encoder, mri_encoder, mri_decoder = load_pretrained_models(ecg_decoder_path, ecg_encoder_path, mri_encoder_path, mri_decoder_path,
                                                                                shared_s=z_s_size, modality_specific_s=z_m_size)
    
    # Load data
    sample_list = get_id_list(data_path, from_file=True, file_name="/home/sana/multimodal/data_list.pkl")
    print("Total number of samples: ", len(sample_list))
    _, valid_loader, test_loader, list_ids = load_data(sample_list, data_path, train_ratio=0.9, test_ratio=0.05)

    rep_disentangler = MultimodalRep(encoders={'input_ecg_rest_median_raw_10_continuous': ecg_encoder, 'input_lax_4ch_heart_center_continuous':mri_encoder}, 
                          decoders={'input_ecg_rest_median_raw_10_continuous': ecg_decoder, 'input_lax_4ch_heart_center_continuous':mri_decoder}, 
                          train_ids=list_ids[0], shared_size=z_s_size, modality_names=modality_names, 
                          z_sizes={'input_ecg_rest_median_raw_10_continuous':z_m_size, 'input_lax_4ch_heart_center_continuous':z_m_size},
                          modality_shapes={'input_ecg_rest_median_raw_10_continuous':(600, 12), 'input_lax_4ch_heart_center_continuous':(96, 96, 50)},
                          mask=True
                          )  

    rep_disentangler.load_from_checkpoint("/home/sana/multimodal/ckpts")
    rep_disentangler._set_trainable_mask(trainable=False)

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
        if i>=15:
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

    # z_s_test = {'input_ecg_rest_median_raw_10_continuous':rep_disentangler.shared_post_mean[:768],
    #              'input_lax_4ch_heart_center_continuous': rep_disentangler.shared_post_mean[:768]}
    # z_m_test = {'input_ecg_rest_median_raw_10_continuous':rep_disentangler.posterior_means[0][:768],
    #             'input_lax_4ch_heart_center_continuous': rep_disentangler.posterior_means[1][:768]}
    # z_s_valid = {'input_ecg_rest_median_raw_10_continuous':rep_disentangler.shared_post_mean[768:1850],
    #              'input_lax_4ch_heart_center_continuous': rep_disentangler.shared_post_mean[768:1850]}
    # z_m_valid = {'input_ecg_rest_median_raw_10_continuous':rep_disentangler.posterior_means[0][768:1850],
    #             'input_lax_4ch_heart_center_continuous': aporep_disentanglerllo_model.posterior_means[1][768:1850]}
    # test_ids = rep_disentangler.train_ids[:768]
    # valid_ids = rep_disentangler.train_ids[768:1850]


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
    mri_pheno = ['LA_2Ch_vol_max', 'LA_2Ch_vol_min', 'LA_4Ch_vol_max', 'LA_4Ch_vol_min', 'LVEDV', 'LVEF',
                 'LVESV', 'LVM', 'LVSV', 'RVEDV', 'RVEF', 'RVESV', 'RVSV']
    phenotype_df = pd.read_csv("/home/sana/tensors_all_union.csv")[ecg_pheno+mri_pheno+['fpath']]
    phenotype_df['fpath'] = phenotype_df['fpath'].astype(int)
    phenotype_df.dropna(inplace=True)
    rep_df = pd.DataFrame({'fpath': test_ids, 'zs_ecg':z_s_test['input_ecg_rest_median_raw_10_continuous'].numpy().tolist(),
                           'zs_mri':z_s_test['input_lax_4ch_heart_center_continuous'].numpy().tolist() ,
                           'zm_ecg':z_m_test['input_ecg_rest_median_raw_10_continuous'].numpy().tolist() ,
                           'zm_mri':z_m_test['input_lax_4ch_heart_center_continuous'].numpy().tolist(),
                            'z_baseline_ecg': baseline_z_test['input_ecg_rest_median_raw_10_continuous'].numpy().tolist(),
                            'z_baseline_mri': baseline_z_test['input_lax_4ch_heart_center_continuous'].numpy().tolist(),
                            'z_merged': z_merged_test.numpy().tolist()}) 
    rep_df_valid = pd.DataFrame({'fpath': valid_ids, 'zs_ecg':z_s_valid['input_ecg_rest_median_raw_10_continuous'].numpy().tolist(),
                           'zs_mri':z_s_valid['input_lax_4ch_heart_center_continuous'].numpy().tolist() ,
                           'zm_ecg':z_m_valid['input_ecg_rest_median_raw_10_continuous'].numpy().tolist() ,
                           'zm_mri':z_m_valid['input_lax_4ch_heart_center_continuous'].numpy().tolist(),
                            'z_baseline_ecg': baseline_z_valid['input_ecg_rest_median_raw_10_continuous'].numpy().tolist(),
                            'z_baseline_mri': baseline_z_valid['input_lax_4ch_heart_center_continuous'].numpy().tolist(),
                            'z_merged': z_merged_valid.numpy().tolist()})
    valid_pheno_df = pd.merge(phenotype_df, rep_df_valid, on='fpath')
    pheno_mean = valid_pheno_df[ecg_pheno+mri_pheno].mean()
    pheno_std = valid_pheno_df[ecg_pheno+mri_pheno].std()
    valid_pheno_df[ecg_pheno+mri_pheno] = (valid_pheno_df[ecg_pheno+mri_pheno]-pheno_mean)/pheno_std
    test_pheno_df = pd.merge(phenotype_df, rep_df, on='fpath')
    test_pheno_df[ecg_pheno+mri_pheno] = (test_pheno_df[ecg_pheno+mri_pheno]-pheno_mean)/pheno_std

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
                             phenotypes=ecg_pheno+mri_pheno, 
                             mask=rep_disentangler.shared_mask.binary_mask)
    print("\nEvaluating shared MRI")
    s2 = phenotype_predictor(np.vstack(valid_pheno_df['zs_mri']), valid_pheno_df, 
                             np.vstack(test_pheno_df['zs_mri']), test_pheno_df,
                             phenotypes=ecg_pheno+mri_pheno, 
                             mask=rep_disentangler.shared_mask.binary_mask)
    print("\nEvaluating modality-specific ECG")
    s3 = phenotype_predictor(np.vstack(valid_pheno_df['zm_ecg']), valid_pheno_df, 
                             np.vstack(test_pheno_df['zm_ecg']), test_pheno_df,
                             phenotypes=ecg_pheno+mri_pheno, 
                             mask=rep_disentangler.modality_masks['input_ecg_rest_median_raw_10_continuous'].binary_mask)
    print("\nEvaluating modality-specific MRI")
    s4 = phenotype_predictor(np.vstack(valid_pheno_df['zm_mri']), valid_pheno_df, 
                             np.vstack(test_pheno_df['zm_mri']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno, 
                             mask=rep_disentangler.modality_masks['input_lax_4ch_heart_center_continuous'].binary_mask)
    print("\nEvaluating baseline ECG")
    b1 = phenotype_predictor(np.vstack(valid_pheno_df['z_baseline_ecg']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_baseline_ecg']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno)
    print("\nEvaluating baseline MRI")
    b2 = phenotype_predictor(np.vstack(valid_pheno_df['z_baseline_mri']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_baseline_mri']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno)
    print("\nEvaluating merged representations")
    m1 = phenotype_predictor(np.vstack(valid_pheno_df['z_merged']), valid_pheno_df, 
                             np.vstack(test_pheno_df['z_merged']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno)
    print("\nEvaluating concatenated baselines")
    m2 = phenotype_predictor(np.concatenate([np.vstack(valid_pheno_df['z_baseline_mri']),np.vstack(valid_pheno_df['z_baseline_ecg'])],-1), 
                             valid_pheno_df, 
                             np.concatenate([np.vstack(test_pheno_df['z_baseline_mri']),np.vstack(test_pheno_df['z_baseline_ecg'])],-1), 
                             test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno)

    print("\nEvaluating concatenated baselines (PCA)")
    m2_pca = phenotype_predictor(np.concatenate([np.vstack(valid_pheno_df['z_baseline_mri']),np.vstack(valid_pheno_df['z_baseline_ecg'])],-1), 
                             valid_pheno_df, 
                             np.concatenate([np.vstack(test_pheno_df['z_baseline_mri']),np.vstack(test_pheno_df['z_baseline_ecg'])],-1), 
                             test_pheno_df, n_pca=rep_disentangler.merged_size,
                             phenotypes=ecg_pheno+mri_pheno) 

    for phenotype in ecg_pheno+mri_pheno:
        labels = ['ecg_zs', 'ecg_zm', 'mri_zs', 'mri_zm', 'ecg rep.', 'mri rep.', 'merged rep.', 'concat. re.', 'concat. rep. pca']
        x = np.arange(len(labels))  # the label locations
        width = 0.4  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained', figsize=(14, 4))
        scores = {'train':(s1[phenotype][0],s3[phenotype][0],s2[phenotype][0],s4[phenotype][0],b1[phenotype][0],b1[phenotype][1],m1[phenotype][0],m2[phenotype][0], m2_pca[phenotype][0]),
         'test':(s1[phenotype][1],s3[phenotype][1],s2[phenotype][1],s4[phenotype][1],b1[phenotype][1],b2[phenotype][1],m1[phenotype][1],m2[phenotype][1], m2_pca[phenotype][1])}

        for attribute, measurement in scores.items():
            offset = width * multiplier
            rounded_list = [round(m*100)/100 for m in measurement]
            rects = ax.bar(x + offset, rounded_list, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('R^2')
        ax.set_title(phenotype)
        ax.set_xticks(x + width, labels, rotation=70)
        # ax.xticks(rotation=70)
        ax.legend(loc='upper left', ncols=4)
        plt.savefig("/home/sana/multimodal/plots/%s.pdf"%phenotype)
        fig.clf()

    # Generalizability test
    train_ours, test_our, train_pca, test_pca = [], [], [], []
    for ratio in [1, 0.8, 0.6, 0.4, 0.2]:
        chopped_df = valid_pheno_df[:int(ratio*len(valid_pheno_df))]
        train_acc, test_acc = phenotype_predictor(np.vstack(chopped_df['z_merged']), chopped_df, 
                             np.vstack(test_pheno_df['z_merged']), test_pheno_df, 
                             phenotypes=ecg_pheno+mri_pheno)
        train_acc_pca, test_acc_pca = phenotype_predictor(np.concatenate([np.vstack(chopped_df['z_baseline_mri']),np.vstack(chopped_df['z_baseline_ecg'])],-1),
                                                        chopped_df, 
                                                        np.concatenate([np.vstack(test_pheno_df['z_baseline_mri']),np.vstack(test_pheno_df['z_baseline_ecg'])],-1), 
                                                        test_pheno_df, n_pca=rep_disentangler.merged_size,
                                                        phenotypes=ecg_pheno+mri_pheno) 
        train_ours.append(train_acc)
        test_our.append(test_acc)
        train_pca.append(train_acc_pca)
        test_pca.append(test_acc_pca)
    
    plt.plot(train_ours, 'r')
    plt.plot(test_our, 'b')
    plt.plot(train_pca, 'g')
    plt.plot(test_pca, 'black')
    plt.savefig("/home/sana/multimodal/plots/generalizability.pdf")

    sys.stdout = orig_stdout
    f.close()


if __name__=="__main__":
    # with open('/home/sana/multimodal/logs/out.txt', 'w') as sys.stdout:
    main()