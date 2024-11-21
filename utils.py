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
from multimodal.representation import MultimodalUKBDataset
# from multimodal.MultiBench.datasets.affect.get_bert_embedding import bert_version_data
import numpy as np
import random

from ml4h.tensormap.ukb.ecg import ecg_rest_median_raw_10
from ml4h.tensormap.ukb.mri import lax_4ch_heart_center

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

def bnorm(x):
    return (x-tf.reduce_mean(x))/tf.math.reduce_std(x)

def load_pretrained_models(decoder_path_1, encoder_path_1, encoder_path_2, decoder_path_2, shared_s, modality_specific_s):
    # custom_dict = {'mish': tfa.activations.mish}
    decoder_1 = tf.keras.models.load_model(decoder_path_1)#, custom_objects=custom_dict)#, compile=False)
    encoder_1 = tf.keras.models.load_model(encoder_path_1)#, custom_objects=custom_dict)#, compile=False)
    encoder_2 = tf.keras.models.load_model(encoder_path_2)#, custom_objects=custom_dict)#, compile=False)
    decoder_2 = tf.keras.models.load_model(decoder_path_2)#, custom_objects=custom_dict)#, compile=False)
    # Randomly initialize decoder parameters
    decoder_1 = randomize_model_weight(decoder_1)
    decoder_2 = randomize_model_weight(decoder_2)
    return decoder_1, encoder_1, encoder_2, decoder_2


def load_data(sample_list, data_paths, n_train=None, train_ratio=0.9, test_ratio=0.05, data='UKB', get_trainset=False):
    if data=='UKB':
        ecg_pheno = ['PQInterval', 'QTInterval','QTCInterval','QRSDuration','RRInterval']
        mri_pheno = ['LA_2Ch_vol_max', 'LA_2Ch_vol_min', 'LA_4Ch_vol_max', 'LA_4Ch_vol_min', 'LVEDV', 'LVEF',
                    'LVESV', 'LVM', 'LVSV', 'RVEDV', 'RVEF', 'RVESV', 'RVSV']
        phenotype_df = pd.read_csv("/home/sana/tensors_all_union.csv")[ecg_pheno+mri_pheno+['fpath']]
        dropfuse_test_ids = pd.read_csv("/home/sana/sample_id_returned_lv_mass.csv")['sample_id']
        dropfuse_test_ids = [str(id)+'.hd5' for id in dropfuse_test_ids]
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
        if n_train is None:
            n_train = int(train_ratio*n_samples)
        n_test = int(test_ratio*n_samples)
        
        train_list = reordered_sample_list[-n_train:]
        # valid_list = reordered_sample_list[n_test:-n_train]
        # test_list = reordered_sample_list[:n_test]
        common_dropfuse = list(set(common_elements)&set(dropfuse_test_ids))
        print(len(list(common_dropfuse)))
        test_list = common_dropfuse[-16:]
        valid_list = common_dropfuse[:-16]
        # valid_list = common_elements[:len(common_elements)//2]
        # test_list = common_elements[len(common_elements)//2:]

        trainset = MultimodalUKBDataset(data_paths, train_list)
        validset = MultimodalUKBDataset(data_paths, valid_list)
        testset = MultimodalUKBDataset(data_paths, test_list)

        train_loader = DataLoader(trainset, batch_size=16, shuffle=False, drop_last=False)
        print("Trainloader length: ", len(trainset), len(validset), len(testset))
        valid_loader = DataLoader(validset, batch_size=32, shuffle=False, drop_last=False)
        test_loader = DataLoader(testset, batch_size=32, shuffle=False, drop_last=False)
        if get_trainset:
            return train_loader, valid_loader, test_loader, (train_list, valid_list, test_list), trainset
        else:
            return train_loader, valid_loader, test_loader, (train_list, valid_list, test_list)

def get_id_list(data_path, data_path_2=None, from_file=False, file_name="data_list.pkl"):
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
                        if all(["ukb_cardiac_mri/cine_segmented_lax_4ch/2" in hd5, "ukb_cardiac_mri/cine_segmented_lax_4ch_annotated_1" in hd5]): 
                            if data_path_2 is None:
                                if all(["ukb_ecg_rest/ecg_rest_text" in hd5, "ukb_ecg_rest/median_I/instance_0" in hd5]):  
                                    sample_list.append(f)
                                else:
                                    empty_files += 1
                            else:
                                try:
                                    with h5py.File(f'{os.path.join(data_path_2, f)}', 'r') as hd5_2:
                                        if "ukb_brain_mri/T1_brain_to_MNI/axial_135/instance_0" in hd5_2:
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

def phenotype_predictor_lf(z_train_1, y_train_1, z_train_2, y_train_2, phenotypes):
    phenotypes_scores = {}
    for pheno in phenotypes:
        if len(np.unique(y_train_1[pheno].to_numpy()))==2:
            auc_test, auc_train = [], []
            kf = KFold(n_splits=4)
            for i, (train_index, test_index) in enumerate(kf.split(z_train_1)):
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
            kf = KFold(n_splits=4)
            for i, (train_index, test_index) in enumerate(kf.split(z_train_1)):
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


def phenotype_predictor(z_train, y_train, z_test, y_test, phenotypes, mask=None, n_pca=None): 
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
            kf = KFold(n_splits=4)
            # kf.get_n_splits(z_train)
            # KFold(n_splits=2, random_state=None, shuffle=False)
            for i, (train_index, test_index) in enumerate(kf.split(z_train)):
                # predictor = RidgeClassifier(alpha=1)
                predictor = SGDClassifier()
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
            # predictor.fit(z_train, y_train[pheno].to_numpy())
            # z_pred_train = predictor.predict(z_train)
            # z_pred = predictor.predict(z_test)
            # auc_test = roc_auc_score(y_test[pheno].to_numpy(), z_pred)
            # auc_train = roc_auc_score(y_train[pheno].to_numpy(), z_pred_train)
            phenotypes_scores[pheno] = [auc_train, auc_test]
        else:
            r2_test, r2_train = [], []
            kf = KFold(n_splits=4)
            # kf.get_n_splits(z_train)
            # KFold(n_splits=2, random_state=None, shuffle=False)
            for i, (train_index, test_index) in enumerate(kf.split(z_train)):
                # predictor = KernelRidge(alpha=1)
                # predictor = SGDRegressor()
                predictor = make_pipeline(StandardScaler(with_mean=True), Ridge(solver='lsqr', max_iter=250000))
                X_train = z_train[train_index]
                X_test = z_train[test_index]
                Y_train = y_train[pheno][train_index]
                Y_test = y_train[pheno][test_index]
                if n_pca is not None and n_pca<X_train.shape[-1]:
                    pca = PCA(n_components=n_pca)
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)
                # predictor = svm.SVR(C=0.1)
                predictor.fit(X_train, Y_train)
                z_pred_train = predictor.predict(X_train)
                z_pred = predictor.predict(X_test)
                r2_test.append(r2_score(Y_test, z_pred))
                r2_train.append(r2_score(Y_train, z_pred_train))
            # predictor = Ridge(alpha=2)
            # predictor = svm.SVR(C=0.1)
            # predictor.fit(z_train, y_train[pheno].to_numpy())
            # z_pred_train = predictor.predict(z_train)
            # z_pred = predictor.predict(z_test)
            # r2_test = r2_score(y_test[pheno].to_numpy(), z_pred)
            # r2_train = r2_score(y_train[pheno].to_numpy(), z_pred_train)
            # # correlations = np.corrcoef(z_test.reshape(len(z_test), -1), y_test[pheno].to_numpy(), rowvar=False)[:len(z_test), -1]
            # correlations = np.corrcoef(z_pred, y_test[pheno].to_numpy(), rowvar=False)
            # print("Pearson correlation of %s: %.5f"%(pheno, correlations.mean()))
            # r2 = predictor.score(z_test, y_test[pheno].to_numpy())
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

# def missing_modality_test(z_train, y_train, z_test, y_test, phenotypes, mask=None, n_pca=None)
#     phenotypes_scores = {}
#     # TODO check for classification setting
#     if mask is not None:
#         if tf.reduce_sum(mask)==0:
#             for pheno in phenotypes:
#                 phenotypes_scores[pheno] = (0,0)
#             return phenotypes_scores
#         z_test = np.take(z_test, np.argwhere(mask==1)[:,0], axis=1)
#         z_train = np.take(z_train, np.argwhere(mask==1)[:,0], axis=1)
#     if n_pca is not None:
#         pca = PCA(n_components=n_pca)
#         z_train = pca.fit_transform(z_train)
#         z_test = pca.transform(z_test)
#     print('Dataset shape: ', z_test.shape, z_train.shape)
#     for pheno in phenotypes:
#         predictor = svm.SVR()
#         predictor.fit(z_train, y_train[pheno].to_numpy())
#         z_pred_train = predictor.predict(z_train)
#         z_pred = predictor.predict(z_test)
#         r2_test = r2_score(y_test[pheno].to_numpy(), z_pred)
#         r2_train = r2_score(y_train[pheno].to_numpy(), z_pred_train)
#         # correlations = np.corrcoef(z_test.reshape(len(z_test), -1), y_test[pheno].to_numpy(), rowvar=False)[:len(z_test), -1]
#         correlations = np.corrcoef(z_pred, y_test[pheno].to_numpy(), rowvar=False)
#         print("Pearson correlation of %s: %.5f"%(pheno, correlations.mean()))
#         phenotypes_scores[pheno] = (r2_train, r2_test)
#     return phenotypes_scores

