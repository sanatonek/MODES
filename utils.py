import os
import pickle
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tensorflow as tf
import numpy as np
import random

from ml4h.tensormap.ukb.ecg import ecg_rest_median_raw_10
from ml4h.tensormap.ukb.mri import lax_4ch_heart_center
from ml4h.tensormap.ukb.mri_brain import t1_mni_slices_128_160

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression, Lasso, SGDRegressor, RidgeClassifier, SGDClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class MultimodalUKBDataset(Dataset):
    def __init__(self, paths, file_names):
        self.paths = paths
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)
    
    def batch_normalize(self,x, std=1):
        return std*(x-torch.mean(x))/torch.std(x)

    def __getitem__(self, idx):
        sample_id = self.file_names[idx].split('.')[0]
        samples = {}
        for modal_name, p in self.paths.items():
            sample_path = os.path.join(p, self.file_names[idx])
            with h5py.File(f'{sample_path}', 'r') as hd5:
                if "ecg" in modal_name:
                    sample = ecg_rest_median_raw_10.normalize(ecg_rest_median_raw_10.tensor_from_file(ecg_rest_median_raw_10, hd5))
                elif "lax" in modal_name:
                    sample = lax_4ch_heart_center.normalize(lax_4ch_heart_center.tensor_from_file(lax_4ch_heart_center, hd5))
                elif "axial" in modal_name:
                    sample = t1_mni_slices_128_160.normalize(t1_mni_slices_128_160.tensor_from_file(t1_mni_slices_128_160, hd5))
            samples[modal_name] = (torch.Tensor(sample))
        return samples, int(sample_id)


def load_pretrained_models(decoder_path_1, encoder_path_1, decoder_path_2, encoder_path_2):
    decoder_1 = tf.keras.models.load_model(decoder_path_1)
    encoder_1 = tf.keras.models.load_model(encoder_path_1)
    encoder_2 = tf.keras.models.load_model(encoder_path_2)
    decoder_2 = tf.keras.models.load_model(decoder_path_2)
    # Randomly initialize decoder parameters
    decoder_1 = randomize_model_weight(decoder_1)
    decoder_2 = randomize_model_weight(decoder_2)
    return decoder_1, encoder_1, decoder_2, encoder_2


def load_data(sample_list, data_paths, n_train=None, test_ratio=0.01, data='UKB', get_trainset=False):
    if data=='UKB':
        ecg_pheno = ['PQInterval', 'QTInterval','QTCInterval','QRSDuration','RRInterval']
        mri_pheno = ['LA_2Ch_vol_max', 'LA_2Ch_vol_min', 'LA_4Ch_vol_max', 'LA_4Ch_vol_min', 'LVEDV', 'LVEF',
                    'LVESV', 'LVM', 'LVSV', 'RVEDV', 'RVEF', 'RVESV', 'RVSV']
        phenotype_df = pd.read_csv("/home/sana/tensors_all_union.csv")[ecg_pheno+mri_pheno+['fpath']]
        phenotype_df.dropna(inplace=True)
        phenotype_df['fpath'] = phenotype_df['fpath'].astype(str) + '.hd5'
        eval_ids = phenotype_df['fpath']  # Patient ids for which we have phenotype labels

        labeled_samples = list(set(sample_list) & set(eval_ids))
        unlabeled_samples = list(set(sample_list) - set(eval_ids))
        if n_train is None:
            n_train = len(unlabeled_samples)
        n_test = int(test_ratio*len(labeled_samples))
        
        train_list = unlabeled_samples[:int(n_train)]
        test_list = labeled_samples[:int(n_test)]
        valid_list = labeled_samples[int(n_test):]

        trainset = MultimodalUKBDataset(data_paths, train_list)
        validset = MultimodalUKBDataset(data_paths, valid_list)
        testset = MultimodalUKBDataset(data_paths, test_list)

        train_loader = DataLoader(trainset, batch_size=64, shuffle=False, drop_last=False)
        valid_loader = DataLoader(validset, batch_size=64, shuffle=False, drop_last=False)
        test_loader = DataLoader(testset, batch_size=64, shuffle=False, drop_last=False)
        print("Trainloader length: ", len(trainset), len(validset), len(testset))
        if get_trainset:
            return train_loader, valid_loader, test_loader, (train_list, valid_list, test_list), trainset
        else:
            return train_loader, valid_loader, test_loader, (train_list, valid_list, test_list)


def get_paired_id_list(data_paths, from_file=False, file_name="data_list.pkl"):
    # This needs to be customized based on dataset
    conditions = {"input_ecg_rest_median_raw_10_continuous":["ukb_ecg_rest/ecg_rest_text","ukb_ecg_rest/median_I/instance_0"],
                  "input_lax_4ch_heart_center_continuous":["ukb_cardiac_mri/cine_segmented_lax_4ch/2", "ukb_cardiac_mri/cine_segmented_lax_4ch_annotated_1"],
                  "input_axial_128_160_continuous":["ukb_brain_mri/T1_brain_to_MNI/axial_135/instance_0"]}
    if from_file:
        with open(file_name, "rb") as f:
            sample_list = pickle.load(f)
    else:
        exlusion_list = ["5833465.hd5", "2144786.hd5"]
        sample_list = []
        empty_files = 0
        data_path_M1 = list(data_paths.values())[0]
        data_path_M2 = list(data_paths.values())[1]
        M1 = list(data_paths.keys())[0]
        M2 = list(data_paths.keys())[1]
        for f in os.listdir(data_path_M1):
            if os.path.isfile(os.path.join(data_path_M1, f)):
                try:
                    with h5py.File(f'{os.path.join(data_path_M1, f)}', 'r') as hd5:
                        if all([cond in hd5 for cond in conditions[M1]]):#["ukb_ecg_rest/ecg_rest_text" in hd5, "ukb_ecg_rest/median_I/instance_0" in hd5]): 
                        # if all(["ukb_cardiac_mri/cine_segmented_lax_4ch/2" in hd5, "ukb_cardiac_mri/cine_segmented_lax_4ch_annotated_1" in hd5]): 
                            if data_path_M1==data_path_M2:
                            # if data_paths[0]==data_paths[1]:
                                # if all(["ukb_ecg_rest/ecg_rest_text" in hd5, "ukb_ecg_rest/median_I/instance_0" in hd5]):  
                                if all([cond in hd5 for cond in conditions[M2]]):#all(["ukb_cardiac_mri/cine_segmented_lax_4ch/2" in hd5, "ukb_cardiac_mri/cine_segmented_lax_4ch_annotated_1" in hd5]): 
                                    sample_list.append(f)
                                else:
                                    empty_files += 1
                            else:
                                try:
                                    with h5py.File(f'{os.path.join(data_path_M2, f)}', 'r') as hd5_2:
                                        if all([cond in hd5_2 for cond in conditions[M2]]):#"ukb_brain_mri/T1_brain_to_MNI/axial_135/instance_0" in hd5_2:
                                            sample_list.append(f)
                                        else:
                                            empty_files += 1
                                except:
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
            if len(batch[index][...,0].shape)==1:
                plt.plot(batch[index][...,0])
            elif len(batch[index][...,0].shape)==2:
                plt.imshow(batch[index][...,0])
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()


def phenotype_predictor_lf(z_train_1, y_train_1, z_train_2, y_train_2, phenotypes, kfold_indices):
    phenotypes_scores = {}
    for pheno in phenotypes:
        if len(np.unique(y_train_1[pheno].to_numpy()))==2:
            auc_test, auc_train = [], []
            for i, (train_index, test_index) in enumerate(kfold_indices):
                predictor_1 = SGDClassifier()
                predictor_2 = SGDClassifier()
                predictor_1.fit(z_train_1[train_index], y_train_1[pheno].to_numpy()[train_index])
                z1_pred_train = predictor_1.predict(z_train_1[train_index])
                z1_pred = predictor_1.predict(z_train_1[test_index])
                predictor_2.fit(z_train_2[train_index], y_train_2[pheno].to_numpy()[train_index])
                z2_pred_train = predictor_2.predict(z_train_2[train_index])
                z2_pred = predictor_2.predict(z_train_2[test_index])
                z_pred_train = (z1_pred_train+z2_pred_train)/2
                z_pred = (z1_pred+z2_pred)/2
                auc_test.append(roc_auc_score(y_train_1[pheno].to_numpy()[test_index], z_pred))
                auc_train.append(roc_auc_score(y_train_1[pheno].to_numpy()[train_index], z_pred_train))
            phenotypes_scores[pheno] = [auc_train, auc_test]
        else:
            r2_test, r2_train = [], []
            for i, (train_index, test_index) in enumerate(kfold_indices):
                predictor_1 = SGDRegressor()
                predictor_2 = SGDRegressor()
                predictor_1.fit(z_train_1[train_index], y_train_1[pheno].to_numpy()[train_index])
                z1_pred_train = predictor_1.predict(z_train_1[train_index])
                z1_pred = predictor_1.predict(z_train_1[test_index])
                predictor_2.fit(z_train_2[train_index], y_train_2[pheno].to_numpy()[train_index])
                z2_pred_train = predictor_2.predict(z_train_2[train_index])
                z2_pred = predictor_2.predict(z_train_2[test_index])
                z_pred_train = (z1_pred_train+z2_pred_train)/2
                z_pred = (z1_pred+z2_pred)/2
                r2_test.append(r2_score(y_train_1[pheno].to_numpy()[test_index], z_pred))
                r2_train.append(r2_score(y_train_1[pheno].to_numpy()[train_index], z_pred_train))
            phenotypes_scores[pheno] = [r2_train, r2_test]
    return phenotypes_scores


def phenotype_predictor(z_train, y_train, z_test, y_test, phenotypes, kfold_indices, mask=None, n_pca=None): 
    phenotypes_scores = {}
    if mask is not None:
        if tf.reduce_sum(mask)==0:
            for pheno in phenotypes:
                phenotypes_scores[pheno] = (0,0)
            return phenotypes_scores
        z_test = np.take(z_test, np.argwhere(mask==1)[:,0], axis=1)
        z_train = np.take(z_train, np.argwhere(mask==1)[:,0], axis=1)
    print("Representation size: ", z_train.shape[-1])
    for pheno in phenotypes:
        if n_pca==0 or z_train.shape[-1]==0:
            phenotypes_scores[pheno] = [[0],[0]]
        if len(np.unique(y_train[pheno].to_numpy()))==2:
            auc_test, auc_train = [], []
            for (train_index, test_index) in kfold_indices:
                predictor = LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', l1_ratio=0.5)
                X_train = z_train[train_index]
                X_test = z_train[test_index]
                Y_train = y_train[pheno].to_numpy()[train_index]
                Y_test = y_train[pheno].to_numpy()[test_index]
                if n_pca is not None and n_pca<X_train.shape[-1]:
                    pca = PCA(n_components=n_pca)
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)
                predictor.fit(X_train, Y_train)
                z_pred_train = predictor.predict(X_train)
                z_pred = predictor.predict(X_test)
                auc_test.append(roc_auc_score(Y_test, z_pred))
                auc_train.append(roc_auc_score(Y_train, z_pred_train))
            phenotypes_scores[pheno] = [auc_train, auc_test]
        else:
            r2_test, r2_train = [], []
            for (train_index, test_index) in kfold_indices:
                predictor = make_pipeline(StandardScaler(with_mean=True), Ridge(solver='lsqr', max_iter=250000))
                X_train = z_train[train_index]
                X_test = z_train[test_index]
                Y_train = y_train[pheno][train_index]
                Y_test = y_train[pheno][test_index]
                if n_pca is not None and n_pca<X_train.shape[-1]:
                    pca = PCA(n_components=n_pca)
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)
                predictor.fit(X_train, Y_train)
                z_pred_train = predictor.predict(X_train)
                z_pred = predictor.predict(X_test)
                r2_test.append(r2_score(Y_test, z_pred))
                r2_train.append(r2_score(Y_train, z_pred_train))
            phenotypes_scores[pheno] = [r2_train, r2_test]
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


def plot_pheno_prediction_performance(phenotypes, perf_results, labels, tag):
    for phenotype in phenotypes:
        print(phenotype)
        x = np.arange(len(labels))  # the label locations
        width = 0.4  # the width of the bars
        multiplier = 0
        fig, ax = plt.subplots(layout='constrained', figsize=(14, 4))
        scores = {'train':[np.mean(s[phenotype][0]) for s in perf_results],
         'test':[np.mean(s[phenotype][1]) for s in perf_results]}
        scores_std = {'train':[np.std(s[phenotype][0]) for s in perf_results],
         'test':[np.std(s[phenotype][1]) for s in perf_results]}
        for attribute, measurement in scores.items():
            offset = width * multiplier
            rounded_list = [round(m*1000)/1000 for m in measurement]
            rects = ax.bar(x + offset, rounded_list, width, label=attribute)
            ax.errorbar(x + offset, rounded_list, yerr=scores_std[attribute], fmt="o", color="black")
            ax.bar_label(rects, padding=3)
            multiplier += 1
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('R^2')
        ax.set_title(phenotype)
        ax.set_xticks(x + width, labels, rotation=70)
        # ax.xticks(rotation=70)
        ax.legend(loc='upper left', ncols=4)
        plt.savefig("/home/sana/multimodal/plots/%s_%s.pdf"%(phenotype,tag))
        fig.clf()


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
