import os
import pickle
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import tensorflow as tf
# import tensorflow_addons as tfa
from multimodal.representation import MultimodalDataset
import numpy as np
import random

from ml4h.tensormap.ukb.ecg import ecg_rest_median_raw_10
from ml4h.tensormap.ukb.mri import lax_4ch_heart_center

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression, Lasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn import svm


def load_pretrained_models(ecg_decoder_path, ecg_encoder_path, mri_encoder_path, mri_decoder_path, shared_s, modality_specific_s):
    # custom_dict = {'mish': tfa.activations.mish}
    ecg_decoder = tf.keras.models.load_model(ecg_decoder_path)#, custom_objects=custom_dict)#, compile=False)
    ecg_encoder = tf.keras.models.load_model(ecg_encoder_path)#, custom_objects=custom_dict)#, compile=False)
    mri_encoder = tf.keras.models.load_model(mri_encoder_path)#, custom_objects=custom_dict)#, compile=False)
    mri_decoder = tf.keras.models.load_model(mri_decoder_path)#, custom_objects=custom_dict)#, compile=False)
    # Randomly initialize decoder parameters
    ecg_decoder = randomize_model_weight(ecg_decoder)
    mri_decoder = randomize_model_weight(mri_decoder)
    return ecg_decoder, ecg_encoder, mri_encoder, mri_decoder


def load_data(sample_list, data_path, train_ratio=0.9, test_ratio=0.05):
    ecg_pheno = ['PQInterval', 'QTInterval','QTCInterval','QRSDuration','RRInterval']
    mri_pheno = ['LA_2Ch_vol_max', 'LA_2Ch_vol_min', 'LA_4Ch_vol_max', 'LA_4Ch_vol_min', 'LVEDV', 'LVEF',
                'LVESV', 'LVM', 'LVSV', 'RVEDV', 'RVEF', 'RVESV', 'RVSV']
    phenotype_df = pd.read_csv("/home/sana/tensors_all_union.csv")[ecg_pheno+mri_pheno+['fpath']]
    phenotype_df.dropna(inplace=True)
    phenotype_df['fpath'] = phenotype_df['fpath'].astype(str) + '.hd5'
    eval_ids = phenotype_df['fpath']  # Patient ids for which we have phenotype labels

    common_elements = list(set(sample_list) & set(eval_ids))
    # sample_list = common_elements
    # print("Total number of ids that we have phenotypes for: ", len(common_elements))
    remaining_elements = list(set(sample_list) - set(eval_ids))
    reordered_sample_list = common_elements + remaining_elements
    # reordered_sample_list = common_elements
    n_samples = len(reordered_sample_list)
    # n_train = int(train_ratio*n_samples)
    n_train = 5000
    n_test = int(test_ratio*n_samples)
    
    train_list = reordered_sample_list[-n_train:]
    # valid_list = reordered_sample_list[n_test:-n_train]
    # test_list = reordered_sample_list[:n_test]
    valid_list = common_elements[:len(common_elements)//2]
    test_list = common_elements[len(common_elements)//2:]

    trainset = MultimodalDataset(data_path, train_list)
    validset = MultimodalDataset(data_path, valid_list)
    testset = MultimodalDataset(data_path, test_list)

    train_loader = DataLoader(trainset, batch_size=16, shuffle=False, drop_last=False)
    print("Trainloader length: ", len(trainset), len(validset), len(testset))
    valid_loader = DataLoader(validset, batch_size=32, shuffle=False, drop_last=False)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False, drop_last=False)
    return train_loader, valid_loader, test_loader, (train_list, valid_list, test_list)

def get_id_list(data_path, from_file=False, file_name="data_list.pkl"):
    if from_file:
        with open(file_name, "rb") as f:
            sample_list = pickle.load(f)
    else:
        exlusion_list = ["5833465.hd5", "2144786.hd5"]
        sample_list = []
        empty_files = 0
        for f in os.listdir(data_path):
            if os.path.isfile(os.path.join(data_path, f)):
                try:
                    with h5py.File(f'{os.path.join(data_path, f)}', 'r') as hd5:
                        # if (('ukb_cardiac_mri' in hd5) and ('ukb_ecg_rest' in hd5)):
                        if all(["ukb_cardiac_mri/cine_segmented_lax_4ch/2" in hd5, "ukb_cardiac_mri/cine_segmented_lax_4ch_annotated_1" in hd5, "ukb_ecg_rest/ecg_rest_text" in hd5, "ukb_ecg_rest/median_I/instance_0" in hd5]):  
                            sample_list.append(f)
                        else:
                            empty_files += 1
                except:
                    empty_files += 1
        for elem in exlusion_list:
            if elem in sample_list:
                sample_list.remove(elem)
        random.shuffle(sample_list)
        with open(file_name, "wb") as f:
            pickle.dump(sample_list, f)
        print("%d empty files"%empty_files)
    return sample_list

def randomize_model_weight(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
            new_weights = layer.kernel_initializer(shape=layer.kernel.shape, dtype=layer.dtype)  # Create new weights
            layer.set_weights([new_weights, layer.bias])  # Set both kernel and bias weights
    return model

def plot_sample(batch, num_cols, num_rows, save_path):
    plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
    for row in range(num_rows):
        for col in range(num_cols):
            index = row * num_cols + col
            plt.subplot(num_rows, num_cols, index + 1)
            if len(batch[index,...,0].shape)==1:
                plt.plot(batch[index,...,0])
            elif len(batch[index,...,0].shape)==2:
                plt.imshow(batch[index,...,0])
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()

def get_ref_sample(path, file_name):
    img_path = os.path.join(path, file_name)
    with h5py.File(f'{img_path}', 'r') as hd5:
        ecg = ecg_rest_median_raw_10.tensor_from_file(ecg_rest_median_raw_10, hd5)
        mri = lax_4ch_heart_center.tensor_from_file(lax_4ch_heart_center, hd5)
    return {"input_ecg_rest_median_raw_10_continuous":ecg[np.newaxis,...], "input_lax_4ch_heart_center_continuous":mri[np.newaxis,...]}


def pca_analysis(data, model, modality_name, percentile=0.25):
    z_s, z_m = model.encode(data)
    z_s, z_m = z_s[modality_name], z_m[modality_name]
    n_top_samples = int(percentile*z_s.shape[0])
    # Find high variance samples based on Zs
    pca_shared = PCA(n_components=2, svd_solver='full')
    z_s_projected = pca_shared.fit_transform(z_s)
    max_pc_zs_inds = z_s_projected[:,0].argsort()[-n_top_samples:]
    min_pc_zs_inds = z_s_projected[:,0].argsort()[:n_top_samples]

    top_pos_samples_s = np.take(data[modality_name], max_pc_zs_inds, 0)#model.decode(np.take(z_s, max_pc_zs_inds, 0), 
                                    #  np.take(z_m, max_pc_zs_inds, 0),  
                                    #  modality_name=modality_name)
    top_neg_samples_s = np.take(data[modality_name], min_pc_zs_inds, 0)#model.decode(np.take(z_s, min_pc_zs_inds, 0), 
                                    #  np.take(z_m, min_pc_zs_inds, 0), 
                                    #  modality_name=modality_name)
    # Find high variance samples based on Zm
    pca_m = PCA(n_components=2, svd_solver='full')
    z_m_projected = pca_m.fit_transform(z_m)
    max_pc_zs_inds = z_m_projected[:,0].argsort()[-n_top_samples:]
    min_pc_zs_inds = z_m_projected[:,0].argsort()[:n_top_samples]
    top_pos_samples_m = np.take(data[modality_name], max_pc_zs_inds, 0)
    top_neg_samples_m = np.take(data[modality_name], min_pc_zs_inds, 0)

    return top_pos_samples_s, top_neg_samples_s, top_pos_samples_m, top_neg_samples_m

def phenotype_predictor(z_train, y_train, z_test, y_test, phenotypes, mask=None):
    phenotypes_scores = {}
    # TODO check for classification setting
    if mask is not None:
        z_test = np.take(z_test, np.argwhere(mask==1)[:,0], axis=1)
        z_train = np.take(z_train, np.argwhere(mask==1)[:,0], axis=1)
    print('Dataset shape: ', z_test.shape, z_train.shape)
    for pheno in phenotypes:
        # predictor = LinearRegression()
        # predictor = KernelRidge()
        predictor = svm.SVR()
        predictor.fit(z_train, y_train[pheno].to_numpy())
        z_pred_train = predictor.predict(z_train)
        z_pred = predictor.predict(z_test)
        print('predicted:', z_pred[:5])
        print('label:', y_test[pheno].to_numpy()[:5])
        # r2_test = (((z_pred - y_test[pheno].to_numpy())**2).mean())/(((y_test[pheno].to_numpy() - y_test[pheno].to_numpy().mean())**2).mean())
        # r2_train = (((z_pred_train - y_train[pheno].to_numpy())**2).mean())/(((y_train[pheno].to_numpy() - y_train[pheno].to_numpy().mean())**2).mean())
        r2_test = r2_score(y_test[pheno].to_numpy(), z_pred)
        r2_train = r2_score(y_train[pheno].to_numpy(), z_pred_train)
        # correlations = np.corrcoef(z_test.reshape(len(z_test), -1), y_test[pheno].to_numpy(), rowvar=False)[:len(z_test), -1]
        correlations = np.corrcoef(z_pred, y_test[pheno].to_numpy(), rowvar=False)
        print("Pearson correlation of %s: %.5f"%(pheno, correlations.mean()))
        # r2 = predictor.score(z_test, y_test[pheno].to_numpy())
        phenotypes_scores[pheno] = (r2_train, r2_test)
    return phenotypes_scores

def cluster_test(df, n_clusters=5, original_modality="ecg"):
    all_labels = {}
    if original_modality=="ecg":
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init="auto").fit(np.vstack(df["z_baseline_ecg"]))
        all_labels["ground_truth"] = kmeans.labels_
        z_m = np.concatenate([np.vstack(df["zm_ecg"]), np.vstack(df["zs_ecg"])], -1)
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init="auto").fit(z_m)
        all_labels["z"] = kmeans.labels_
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init="auto").fit(np.vstack(df["zm_mri"]))
        all_labels["zm_other"] = kmeans.labels_
    elif original_modality=="mri":
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init="auto").fit(np.vstack(df["z_baseline_mri"]))
        all_labels["ground_truth"] = kmeans.labels_
        z_m = np.concatenate([np.vstack(df["zm_mri"]), np.vstack(df["zs_mri"])], -1)
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init="auto").fit(z_m)
        all_labels["z"] = kmeans.labels_
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init="auto").fit(np.vstack(df["zm_ecg"]))
        all_labels["zm_other"] = kmeans.labels_
        
    s1 = hungarian_match(all_labels["ground_truth"], all_labels["z"], n_clusters)
    s2 = hungarian_match(all_labels["ground_truth"], all_labels["zm_other"], n_clusters)
    return [s1, s2]


def hungarian_match(y, y_hat, n_clusters):   
    cm = confusion_matrix(y, y_hat, labels=np.arange(n_clusters))  # the ij'th element is the number of class i predicted as class
    row_ind, col_ind = linear_assignment(cm, maximize=True)
    mapping = {}
    for true_labels in np.unique(y):
        mapping[int(true_labels)] = col_ind[int(true_labels)]
    mapped_scores = np.copy(y_hat)
    for (gt_z, pred_z) in mapping.items():
        mapped_scores[y_hat == pred_z] = gt_z
    return (mapped_scores==y).mean() 

