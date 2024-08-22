import torch
from torch.utils.data import Dataset
import tensorflow as tf
from tensorflow import keras 
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
# from tf_agents.distributions.gumbel_softmax import GumbelSoftmax 
sns.set_theme()
# from keras.utils.layer_utils import count_params
from ml4h.tensormap.ukb.ecg import ecg_rest_median_raw_10
from ml4h.tensormap.ukb.mri import lax_4ch_heart_center
import h5py
#for TF2
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import logistic



class MultimodalRep():
    def __init__(self, encoders, z_sizes, shared_size, modality_names, modality_shapes, train_ids=None, n_train=None, decoders=None, mask=False, beta=1):
        super(MultimodalRep, self).__init__()
        self.mask = mask
        self.z_sizes = z_sizes
        self.shared_size = shared_size
        if n_train is None:
            self.n_train = len(train_ids)
        else:
            self.n_train = n_train
        self.n_modalitites = len(modality_names)
        self.modality_names = modality_names
        self.modality_shapes = modality_shapes
        self.beta = beta
        self.merged_size = None
        
        self.posterior_means = {}
        for mod, z_size in z_sizes.items():
            self.posterior_means[mod] = tf.Variable(tf.random.normal((self.n_train,(z_size))))
        self.shared_post_mean = tf.Variable(tf.random.normal((self.n_train,(shared_size))), trainable=True)
        self.shared_mask = LearnableMask(shared_size)
        self.modality_masks = {}
        for modality_name, z_size in self.z_sizes.items():
            self.modality_masks[modality_name] = LearnableMask(z_size)
        
        # models
        self.encoders, self.decoders = {}, {}
        self.reference_encoding = {}
        for model_name in modality_names:
            encoder, decoder = self._add_disentangled_block(encoders[model_name], 
                                                            modality_shapes[model_name], z_s_size=shared_size, 
                                                            z_m_size=z_sizes[model_name], trainable=True)
            self.encoders[model_name] = encoder
            self.decoders[model_name] = decoder

    def _set_trainable_mask(self, trainable):
        for modality_name, z_size in self.z_sizes.items():
            self.modality_masks[modality_name].trainable = trainable
        self.shared_mask.trainable = trainable

    def _add_disentangled_block(self,encoder, input_size, z_s_size, z_m_size, trainable=False):
        for layer in encoder.layers:
            layer.trainable = False
        inputs = keras.Input(shape=input_size, dtype=tf.float32)
        x = encoder(inputs, training=False)
        fc1 = keras.layers.Dense(z_m_size+z_s_size, name='l1', activation='relu')(x[0])
        output1 = keras.layers.Dense(z_m_size, name='modality')(fc1)
        output2 = keras.layers.Dense(z_s_size, name='shared')(fc1)
        encoder = keras.models.Model(inputs=inputs, outputs=[output1, output2, x[0]])
        reps = keras.layers.Input(shape=(z_m_size+z_s_size,), dtype=tf.float32) 
        fc1 = keras.layers.Dense(512, name='l1', activation='relu')(reps)
        reconst = keras.layers.Dense(256, name='fusion')(fc1)
        decoder = tf.keras.models.Model(inputs=(reps), outputs=reconst)
        return encoder, decoder

    def load_from_checkpoint(self, ckpt_path):
        for modal_name in (self.modality_names):
            self.encoders[modal_name].load_weights(os.path.join(ckpt_path, (modal_name+"_encoder.weights.h5")))
            self.decoders[modal_name].load_weights(os.path.join(ckpt_path, (modal_name+"_decoder.weights.h5")))
            self.posterior_means[modal_name] = np.load(os.path.join(ckpt_path, "modality_z_%s.npy"%modal_name))
            self.modality_masks[modal_name].mask = np.load(os.path.join(ckpt_path, "mask_%s.npy"%modal_name))
            self.modality_masks[modal_name].binary_mask = np.load(os.path.join(ckpt_path, "mask_%s_bin.npy"%modal_name))
            print(tf.sigmoid(self.modality_masks[modal_name].mask))
            print(self.modality_masks[modal_name].test())
            print('Percentage of selected features for %s: '%modal_name, tf.math.reduce_sum(self.modality_masks[modal_name].binary_mask)/self.z_sizes[modal_name])
        self.shared_post_mean = np.load('/home/sana/multimodal/ckpts/shared_z.npy')
        self.shared_mask.mask = np.load('/home/sana/multimodal/ckpts/shared_mask.npy')
        self.shared_mask.binary_mask = np.load('/home/sana/multimodal/ckpts/shared_mask_bin.npy')
        self._set_trainable_mask(trainable=False)
        print('Percentage of selected features for shared z: ', tf.math.reduce_sum(self.shared_mask.binary_mask)/self.shared_size)
        print(tf.sigmoid(self.shared_mask.mask))
        print(self.shared_mask.test())
    
    def train(self, trainloader, lr_enc=0.001, lr_dec=0.001, epochs_enc=20, epochs_dec=20):
        dec_loss, enc_loss, shared_loss, m_loss = [], [], [], []
        shared_mask = []
        modality_specific_masks = {mod: [] for mod in self.modality_names}
        n_rounds = 20
        initial_temperature = 1
        anneal_rate = -np.log(0.2 / initial_temperature) / (n_rounds-1)
        optimizer_1 = tf.keras.optimizers.Adam(learning_rate=lr_dec, clipvalue=1)
        optimizer_2 = tf.keras.optimizers.Adam(learning_rate=lr_dec, clipvalue=1)
        optimizer_3 = tf.keras.optimizers.Adam(learning_rate=lr_dec, clipvalue=1)
        optimizer_4 = tf.keras.optimizers.Adam(learning_rate=lr_enc, clipvalue=1)
        for iter_round in range(n_rounds):
            # dec_loss.extend(self.train_mask(trainloader, lr_dec, epochs_dec, optimizer_1))
            m_loss.extend(self.optimize_modality_latent(trainloader, lr_dec, epochs_dec, optimizer_2))
            shared_loss.extend(self.optimize_shared_latent(trainloader, lr_dec, epochs_dec, optimizer_3))
            enc_loss.extend(self.train_encoder(trainloader, lr_enc, epochs_enc, optimizer_4))
            # Annealing temperature
            for modality_name, z_size in self.z_sizes.items():
                self.modality_masks[modality_name].temperature = tf.convert_to_tensor(initial_temperature * np.exp(-anneal_rate*(iter_round)), dtype=tf.float32)
                print("Modality mask: ", self.modality_masks[modality_name].mask[:5])
                self.shared_mask.temperature = tf.convert_to_tensor(initial_temperature * np.exp(-anneal_rate*(iter_round)), dtype=tf.float32)
            print("Shared mask: ", self.shared_mask.mask[:5])
            print("Temperature: ", self.modality_masks[modality_name].temperature)
            # Track mask updates
            shared_mask.append(self.shared_mask.binary_mask.numpy())
            for mod in self.modality_names:
                modality_specific_masks[mod].append(self.modality_masks[mod].binary_mask.numpy())
        # Plot masks over iterations 
        f, axs = plt.subplots(self.n_modalitites+1, 1)
        for nn, ax in enumerate(axs):
            if nn == self.n_modalitites:
                sns.heatmap(np.array(shared_mask), ax=ax)
                ax.set_xlabel('Shared mask')
            else:
                sns.heatmap(np.array(modality_specific_masks[self.modality_names[nn]]), ax=ax)
                ax.set_xlabel(self.modality_names[nn])
        plt.savefig("/home/sana/multimodal/plots/masks.pdf")
        return dec_loss, enc_loss, shared_loss, m_loss

    def infoNCE(self, z_orig, reconst):
        dot_product = tf.matmul(reconst, z_orig, transpose_b=True)
        reconst_norm = tf.norm(reconst, axis=1, keepdims=True)
        z_orig_norm = tf.norm(z_orig, axis=1, keepdims=True)
        cosine_similarity_matrix = dot_product / (reconst_norm * tf.transpose(z_orig_norm) + 1e-10)
        positive_similarities = tf.linalg.diag_part(cosine_similarity_matrix)
        sum_of_similarities = tf.reduce_sum(cosine_similarity_matrix, axis=1)

        infonce_loss = -tf.math.log(positive_similarities / sum_of_similarities + 1e-10)
        return tf.reduce_mean(infonce_loss)

    # def train_mask(self, trainloader, lr, n_epochs, optimizer):
    #     print(">>>>> Training the decoder ...")
    #     # loss_fn = tf.keras.losses.MeanSquaredError()
    #     # optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=1)
    #     self._set_trainable_mask(trainable=False)
    #     loss_trend = []   
    #     for epoch in range(n_epochs):
    #         batch_ind = 0
    #         for data,_ in trainloader:
    #             loss = 0#(self.beta*(self.shared_mask.l1())+10*(self.shared_mask.entropy_regularization())) if self.mask else 0
    #             trainable_var = []#[self.shared_mask.mask] if self.mask else []
    #             with tf.GradientTape(persistent=True) as tape:
    #                 for m_ind, (mod, z) in enumerate(self.posterior_means.items()):
    #                     x_m = tf.convert_to_tensor(data[mod])
    #                     batch_size = x_m.shape[0]
    #                     _, _, z_orig = self.encoders[mod](x_m)
    #                     noise = tf.random.normal(shape=(batch_size ,self.z_sizes[mod]+self.shared_size), mean=0.0, stddev=0.05)
    #                     prev_modality = self.modality_names[m_ind-1 if m_ind>0 else self.n_modalitites-1]
    #                     _, z_s_other, z_other = self.encoders[prev_modality](tf.convert_to_tensor(data[prev_modality]))
    #                     # z_s_other = self.shared_mask(z_s_other)
    #                     z_s_other = self.shared_mask(z_other)
    #                     z_m = self.modality_masks[mod](z[batch_ind:batch_ind+batch_size])
    #                     # trainable_var.extend([self.modality_masks[mod].mask])
    #                     # loss += self.beta*(self.modality_masks[mod].l1())
    #                     # loss += 10*(self.modality_masks[mod].entropy_regularization())
                        
    #                     reconst = self.decoders[mod](tf.concat([z_m, z_s_other], -1)+noise)
    #                     # cosine_similarity_matrix = tf.matmul(tf.nn.l2_normalize(reconst, axis=1), 
    #                     #                                      tf.nn.l2_normalize(z_orig, axis=1), transpose_b=True)
    #                     # positive_similarities = tf.linalg.diag_part(cosine_similarity_matrix)
    #                     # sum_of_similarities = tf.reduce_sum(cosine_similarity_matrix, axis=1)

    #                     dot_product = tf.matmul(reconst, z_orig, transpose_b=True)
    #                     reconst_norm = tf.norm(reconst, axis=1, keepdims=True)
    #                     z_orig_norm = tf.norm(z_orig, axis=1, keepdims=True)
    #                     cosine_similarity_matrix = dot_product / (reconst_norm * tf.transpose(z_orig_norm) + 1e-10)
    #                     positive_similarities = tf.linalg.diag_part(cosine_similarity_matrix)
    #                     sum_of_similarities = tf.reduce_sum(cosine_similarity_matrix, axis=1)

    #                     infonce_loss = -tf.math.log(positive_similarities / sum_of_similarities + 1e-10)
    #                     loss += tf.reduce_mean(infonce_loss)


    #                     # loss += loss_fn(reconst, z_orig)
    #                     trainable_var.extend(self.decoders[mod].trainable_variables)
    #             gradients = tape.gradient(loss, trainable_var)
    #             optimizer.apply_gradients(zip(gradients, trainable_var))
    #             batch_ind += batch_size   
    #             del tape       
    #         np.save('/home/sana/multimodal/ckpts/shared_mask', self.shared_mask.mask.numpy())
    #         np.save('/home/sana/multimodal/ckpts/shared_mask_bin', self.shared_mask.binary_mask.numpy())
    #         for (mod, decoder) in self.decoders.items():
    #             decoder.save_weights("/home/sana/multimodal/ckpts/%s_decoder.weights.h5"%(mod))
    #             np.save('/home/sana/multimodal/ckpts/mask_%s'%mod, self.modality_masks[mod].mask.numpy())
    #             np.save('/home/sana/multimodal/ckpts/mask_%s_bin'%mod, self.modality_masks[mod].binary_mask.numpy())
    #         loss_trend.append(loss.numpy())
    #     return loss_trend

    
    def optimize_modality_latent(self, trainloader, lr, n_epochs, optimizer):
        print(">>>>> Training the modality-specific latent ...")
        # loss_fn = tf.keras.losses.MeanSquaredError()
        loss_trend = []  
        for modality_name, z_size in self.z_sizes.items():
            self.modality_masks[modality_name].trainable = False
        self.shared_mask.trainable = True
        for epoch in range(n_epochs):
            batch_ind = 0
            for data,_ in trainloader:
                loss = (self.beta*(self.shared_mask.l1())+10*(self.shared_mask.entropy_regularization())) if self.mask else 0
                trainable_var = [self.shared_mask.mask] if self.mask else []
                with tf.GradientTape() as tape:
                    for m_ind, (mod, z) in enumerate(self.posterior_means.items()):
                        x_m = tf.convert_to_tensor(data[mod])
                        batch_size = x_m.shape[0]
                        noise = tf.random.normal(shape=(batch_size ,self.z_sizes[mod]+self.shared_size), mean=0.0, stddev=0.05)
                        prev_modality = self.modality_names[m_ind-1 if m_ind>0 else self.n_modalitites-1]
                        _,z_s_other, z_other = self.encoders[prev_modality](tf.convert_to_tensor(data[prev_modality]))
                        _, _, z_orig = self.encoders[mod](x_m)
                        if self.mask:
                            # z_s_other = self.shared_mask(z_s_other)
                            z_s_other = self.shared_mask(z_other)
                            z_m = self.modality_masks[mod](z[batch_ind:batch_ind+batch_size])
                        else:
                            z_m = z[batch_ind:batch_ind+batch_size]
                        # trainable_var.extend([self.modality_masks[mod].mask])
                        # loss += self.beta*(self.modality_masks[mod].l1())
                        # loss += 10*(self.modality_masks[mod].entropy_regularization())

                        reconst = self.decoders[mod](tf.concat([z_m, z_s_other], -1)+noise)
                        loss += self.infoNCE(z_orig, reconst)#tf.reduce_mean((z_orig-reconst)**2)
                        loss += 0.01*tf.reduce_mean(self.posterior_means[mod][batch_ind:batch_ind+batch_size]**2)
                        trainable_var.extend([self.posterior_means[mod]])
                        trainable_var.extend(self.decoders[mod].trainable_variables)
                gradients = tape.gradient(loss, trainable_var)
                optimizer.apply_gradients(zip(gradients, trainable_var))
                batch_ind += batch_size         
                del tape   
            # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.numpy()}")
            loss_trend.append(loss.numpy())   
            np.save('/home/sana/multimodal/ckpts/shared_mask', self.shared_mask.mask.numpy())
            np.save('/home/sana/multimodal/ckpts/shared_mask_bin', self.shared_mask.binary_mask.numpy())
            for mod, zm in self.posterior_means.items():
                self.decoders[mod].save_weights("/home/sana/multimodal/ckpts/%s_decoder.weights.h5"%(mod))
                np.save("/home/sana/multimodal/ckpts/modality_z_%s"%mod, zm.numpy())
        return loss_trend
    
    def optimize_shared_latent(self, trainloader, lr, n_epochs, optimizer):
        print(">>>>> Training the shared latent representation ...")
        # loss_fn = tf.keras.losses.MeanSquaredError()
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=1)
        loss_trend = []
        for modality_name, z_size in self.z_sizes.items():
            self.modality_masks[modality_name].trainable = True
        self.shared_mask.trainable = False
        for epoch in range(n_epochs):
            batch_ind = 0
            for data,_ in trainloader:
                loss = 0#(self.beta*(self.shared_mask.l1())+10*(self.shared_mask.entropy_regularization())) if self.mask else 0
                trainable_var = [self.shared_post_mean] #+ [self.shared_mask.mask] 
                with tf.GradientTape() as tape:
                    for (mod, z) in self.posterior_means.items():
                        x_m = tf.convert_to_tensor(data[mod])
                        batch_size = x_m.shape[0]
                        noise = tf.random.normal(shape=(batch_size ,self.z_sizes[mod]+self.shared_size), mean=0.0, stddev=0.05)
                        _, _, z_orig = self.encoders[mod](x_m)
                        if self.mask:
                            z_m = self.modality_masks[mod](z[batch_ind:batch_ind+batch_size])
                            z_s = self.shared_mask(self.shared_post_mean[batch_ind:batch_ind+batch_size])
                        else:
                            z_m = z[batch_ind:batch_ind+batch_size]
                            z_s = self.shared_post_mean[batch_ind:batch_ind+batch_size]
                        reconst = self.decoders[mod](tf.concat([z_m, z_s], -1)+noise)
                        loss += self.infoNCE(z_orig, reconst)#tf.reduce_mean((z_orig-reconst)**2)
                        trainable_var.extend([self.modality_masks[mod].mask])
                        trainable_var.extend(self.decoders[mod].trainable_variables)
                        loss += self.beta*(self.modality_masks[mod].l1()) + 10*(self.modality_masks[mod].entropy_regularization())
                    loss += 0.01*tf.reduce_mean(self.shared_post_mean[batch_ind:batch_ind+batch_size]**2)
                gradients = tape.gradient(loss, trainable_var)
                optimizer.apply_gradients(zip(gradients, trainable_var))
                batch_ind += batch_size  
                del tape          
            # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.numpy()}")
            loss_trend.append(loss.numpy())
        for mod, zm in self.posterior_means.items():
            self.decoders[mod].save_weights("/home/sana/multimodal/ckpts/%s_decoder.weights.h5"%(mod))
            np.save('/home/sana/multimodal/ckpts/mask_%s'%mod, self.modality_masks[mod].mask.numpy())
            np.save('/home/sana/multimodal/ckpts/mask_%s_bin'%mod, self.modality_masks[mod].binary_mask.numpy())
        np.save("/home/sana/multimodal/ckpts/shared_z", self.shared_post_mean.numpy())
        print('Shared rep. Loss: ', np.mean(loss_trend))
        return loss_trend
    
    def train_encoder(self, trainloader, lr, n_epochs, optimizer):
        print(">>>>> Training the encoder ...")
        # loss_fn = keras.losses.MeanSquaredError()
        # optimizer = keras.optimizers.Adam(learning_rate=lr, clipvalue=1)
        loss_trend = []
        self._set_trainable_mask(trainable=False)
        for epoch in range(n_epochs):
            batch_ind = 0
            for data,_ in trainloader:
                loss = 0
                trainable_var = []
                with tf.GradientTape() as tape:
                    for (mod, z) in self.posterior_means.items():
                        x_m = tf.convert_to_tensor(data[mod])
                        batch_size = x_m.shape[0]
                        z_m, z_s, _ = self.encoders[mod](x_m)
                        shared_mse = (self.shared_post_mean[batch_ind:batch_ind+batch_size] - z_s)**2
                        mod_mse = (z[batch_ind:batch_ind+batch_size]-z_m)**2
                        shared_loss = tf.reduce_mean(self.shared_mask(shared_mse))
                        modality_loss = tf.reduce_mean(self.modality_masks[mod](mod_mse))
                        loss += 2*shared_loss + modality_loss
                        trainable_var.extend(self.encoders[mod].trainable_variables)
                gradients = tape.gradient(loss, trainable_var)
                optimizer.apply_gradients(zip(gradients, trainable_var))
                del tape
                batch_ind += batch_size 
            loss_trend.append(loss.numpy())
            for (mod, encoder) in self.encoders.items():
                encoder.save_weights("/home/sana/multimodal/ckpts/%s_encoder.weights.h5"%(mod))
        print('Encoder Loss: ', np.mean(loss_trend))
        return loss_trend
    
    def merge_representations(self, x_batch):
        self._set_trainable_mask(trainable=False)
        z_m_all, z_s_all = [], []
        for mod, encoder in (self.encoders).items():
            if x_batch[mod] is not None:  # No missing modality
                x_m = tf.convert_to_tensor(x_batch[mod])
                z_m, z_s, _ = encoder(x_m)
                modality_mask = self.modality_masks[mod].binary_mask
                z_m = np.take(z_m, np.argwhere(modality_mask==1)[:,0], axis=1)
                z_m_all.append(z_m)
                shared_mask = self.shared_mask.binary_mask
                z_s = np.take(z_s, np.argwhere(shared_mask==1)[:,0], axis=1)
                z_s_all.append(z_s)
        z = tf.concat(z_m_all+[tf.reduce_mean(z_s_all, 0)], axis=-1)
        # z = tf.concat(z_m_all+[z_s_all[1]], axis=-1)
        self.merged_size = z.shape[-1]
        return z

    def encode(self, x_batch, remove_masks=False):
        self._set_trainable_mask(trainable=False)
        z_m_all, z_s_all = {}, {}
        for mod, encoder in (self.encoders).items():
            x_m = tf.convert_to_tensor(x_batch[mod])
            z_m, z_s, _ = encoder(x_m)
            if self.mask:
                if remove_masks:
                    z_s_all[mod] = self.shared_mask(z_s)
                    z_m_all[mod] = self.modality_masks[mod](z_m)
                else:
                    z_s_all[mod] = self.shared_mask(z_s)
                    z_m_all[mod] = self.modality_masks[mod](z_m)
            else:
                z_s_all[mod] = z_s
                z_m_all[mod] = z_m
        return z_s_all, z_m_all

    def decode(self, z_s_all, z_m_all, x=None, modality_name=None):
        self._set_trainable_mask(trainable=False)
        if modality_name is None:
            x_recon = {}
            for m_ind, (mod, decoder) in enumerate((self.decoders).items()):
                noise = tf.random.normal(shape=(z_m_all[mod].shape[0] ,self.z_sizes[mod]+self.shared_size), mean=0.0, stddev=0.2)
                if x is None: # Decode using the shared representation of the encoder
                    if self.mask:
                        x_recon[mod] = decoder(tf.concat([self.modality_masks[mod](z_m_all[mod]), 
                                                          self.shared_mask(z_s_all[mod])], -1)+noise)
                    else:
                        x_recon[mod] = decoder(tf.concat([z_m_all[mod], z_s_all[mod]], -1)+noise)
                else: # Decode using the data from the other modality
                    prev_modality = self.modality_names[m_ind-1 if m_ind>0 else self.n_modalitites-1]
                    _,conditional_input,_ = self.encoders[prev_modality](tf.convert_to_tensor(x[prev_modality]))
                    if self.mask:
                        x_recon[mod] = decoder(tf.concat([self.modality_masks[mod](z_m_all[mod]), 
                                                          self.shared_mask(conditional_input)], -1)+noise)
                    else:
                        x_recon[mod] = decoder(tf.concat([z_m_all[mod], conditional_input], -1)+noise)
        else:
            decoder = self.decoders[modality_name]
            noise = tf.random.normal(shape=(z_m_all.shape[0] ,self.z_sizes+self.shared_size), mean=0.0, stddev=0.2)
            if x is None:
                if self.mask:
                    x_recon[mod] = decoder(tf.concat([self.modality_masks[modality_name](z_m_all), 
                                                    self.shared_mask(z_s_all)], -1)+noise)
                else:
                    x_recon[mod] = decoder(tf.concat([z_m_all, z_s_all], -1)+noise)
            else:
                m_ind = self.modality_names.index(modality_name)
                other_m = self.modality_names[m_ind-1 if m_ind>0 else self.n_modalitites-1]
                _,conditional_input,_ = self.encoders[other_m](tf.convert_to_tensor(x[other_m]))
                if self.mask:
                    x_recon[mod] = decoder(tf.concat([self.modality_masks[modality_name](z_m_all), 
                                                    self.shared_mask(conditional_input)], -1)+noise)
                else:
                    x_recon[mod] = decoder(tf.concat([z_m_all, conditional_input], -1)+noise)
        return x_recon


class AffectDataset(Dataset):
    def __init__(self, path, file_names):
        self.path = path
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.file_names[idx])
        sample_id = self.file_names[idx].split('.')[0]
        with h5py.File(f'{img_path}', 'r') as hd5:
            ecg = ecg_rest_median_raw_10.tensor_from_file(ecg_rest_median_raw_10, hd5)
            mri = lax_4ch_heart_center.tensor_from_file(lax_4ch_heart_center, hd5)
        return {"input_ecg_rest_median_raw_10_continuous":torch.Tensor(ecg), "input_lax_4ch_heart_center_continuous":torch.Tensor(mri)}, int(sample_id)

class LearnableMask(keras.layers.Layer):
    def __init__(self, input_dim, temperature=1.0, straight_through=True):
        # TODO: Understand the Nan problem with temperature
        super(LearnableMask, self).__init__()
        self.mask = tf.Variable(initial_value=tf.zeros(input_dim), trainable=True)
        self.temperature = temperature
        self.straight_through = straight_through
        self.trainable = True
        self.binary_mask = tf.round(self.gumbel_softmax(self.mask))

    def mask_size(self):
        return tf.reduce_sum(self.binary_mask)

    def l1(self):
        # mask = self.gumbel_softmax(self.mask)
        mask = tf.sigmoid(self.mask)
        return tf.reduce_sum(mask)

    def entropy_regularization(self):
        # Clip values to avoid log(0)
        mask = tf.sigmoid(self.mask)
        # Compute entropy for each mask value
        entropy = -mask * tf.math.log(mask) - (1 - mask) * tf.math.log(1 - mask)
        # Sum entropy across all mask values
        entropy_loss = tf.reduce_sum(entropy)
        # Apply the regularization term
        return entropy_loss
    
    def gumbel_softmax(self, logits, n_samples=0.2):
        probs = tf.sigmoid(logits)
        probs = tf.clip_by_value(probs, 1e-3, 1 - 1e-3) 
        # dist = logistic.Logistic(logits/self.temperature, 1./self.temperature)
        # samples = dist.sample()
        # print('********', logits)
        distribution = tfp.distributions.RelaxedBernoulli(self.temperature, probs=probs, allow_nan_stats=False)
        # print(distribution.sample())
        sample = distribution.sample()
        return tf.clip_by_value(sample, 1e-12, 1.)
    
    def test(self):
        return self.gumbel_softmax(self.mask)

    def call(self, x):
        # Gumbel-Softmax Sampling
        # mask = tf.nn.gumbel_softmax(self.mask, tau=self.temperature, hard=self.straight_through)
        if self.trainable is False:
            return self.binary_mask * x
        else:
            mask = self.gumbel_softmax(self.mask)
            self.binary_mask = tf.round(mask)
            return x * mask  # Element-wise multiplication
    
    # def get_binary_mask(self):
    #     bin_mask = tf.reduce_mean(self.gumbel_softmax(self.mask, n_samples=1), 0)
    #     bin_mask = tf.round(bin_mask)
    #     return bin_mask
    
