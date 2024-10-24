import torch
from torch.utils.data import Dataset
import tensorflow as tf
from tensorflow import keras 
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import gc
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
    def __init__(self, encoders, decoders, train_ids, z_sizes, shared_size, modality_names, modality_shapes, mask=False, beta=1):
        super(MultimodalRep, self).__init__()
        self.mask = mask
        self.z_sizes = z_sizes
        self.shared_size = shared_size
        self.train_ids = [int(train_ids[idx].split('.')[0]) for idx in range(len(train_ids))]
        self.n_train = len(train_ids)
        self.n_modalitites = len(modality_names)
        self.modality_names = modality_names
        self.modality_shapes = modality_shapes
        self.beta = beta
        self.merged_size = None
        # self.z_sizes_masked = z_sizes.copy()
        # self.shared_size_masked = shared_size.copy()
        
        self.posterior_means = {}
        for mod, z_size in z_sizes.items():
            self.posterior_means[mod] = tf.Variable(tf.random.normal((self.n_train,(z_size))))
        # self.posterior_means = [tf.Variable(tf.random.normal((self.n_train,(z_size))), trainable=True) for z_size in z_sizes.values()]
        self.shared_post_mean = tf.Variable(tf.random.normal((self.n_train,(shared_size))), trainable=True)
        self.shared_mask = LearnableMask(shared_size, init_mask_val=(1 if self.mask else 50))
        self.modality_masks = {}
        for modality_name, z_size in self.z_sizes.items():
            self.modality_masks[modality_name] = LearnableMask(z_size, init_mask_val=(0 if self.mask else 50))
        
        # models
        self.encoders, self.decoders = {}, {}
        self.reference_encoding = {}
        for model_name in modality_names:
            encoder, decoder = self._add_disentangled_block(encoders[model_name], decoders[model_name], 
                                                            modality_shapes[model_name], z_s_size=shared_size, 
                                                            z_m_size=z_sizes[model_name], trainable=True)
            self.encoders[model_name] = encoder
            self.decoders[model_name] = decoder

    def _set_trainable_mask(self, trainable):
        for modality_name, z_size in self.z_sizes.items():
            self.modality_masks[modality_name].trainable = trainable
        self.shared_mask.trainable = trainable

    def _add_disentangled_block(self,encoder, decoder, input_size, z_s_size, z_m_size, trainable=False):
        # if not trainable:
        #     for k,v in encoder._get_trainable_state().items():
        #         k.trainable = False
        #     for k,v in decoder._get_trainable_state().items():
        #         k.trainable = False
        for layer in encoder.layers:
            layer.trainable = False
        inputs = keras.Input(shape=input_size, dtype=tf.float32)
        x = encoder(inputs, training=False)
        # fc1 = keras.layers.Dense(512, name='l1', activation='relu')(x[0])
        output1 = keras.layers.Dense(z_m_size, name='modality')(x[0])
                                    #  kernel_initializer=keras.initializers.Constant(value=0.01))(x[0])
        output2 = keras.layers.Dense(z_s_size, name='shared')(x[0])
                                    #  kernel_initializer=keras.initializers.Constant(value=0.01))(x[0])
        # encoder = keras.Model(inputs=inputs, outputs=[output1, output2, x])
        encoder = keras.models.Model(inputs=inputs, outputs=[output1, output2, x[0]])
        # reps = keras.Input(shape=(z_m_size+z_s_size)) 
        reps = keras.layers.Input(shape=(z_m_size+z_s_size,), dtype=tf.float32) 
        x1 = keras.layers.Dense(256, name='fusion')(reps)
        reconst = decoder(x1)
        # decoder = tf.keras.Model(inputs=(reps), outputs=reconst)
        decoder = tf.keras.models.Model(inputs=(reps), outputs=reconst[0])
        # Randomize decoder weights
        weights = decoder.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        decoder.set_weights(weights)
        return encoder, decoder

    def load_from_checkpoint(self, ckpt_path):
        for modal_name in (self.modality_names):
            self.encoders[modal_name].load_weights(os.path.join(ckpt_path, (modal_name+"_encoder.weights.h5")))
            self.decoders[modal_name].load_weights(os.path.join(ckpt_path, (modal_name+"_decoder.weights.h5")))
            self.posterior_means[modal_name] = np.load(os.path.join(ckpt_path, "modality_z_%s.npy"%modal_name))
            self.modality_masks[modal_name].mask_logits = np.load(os.path.join(ckpt_path, "mask_%s.npy"%modal_name))
            self.modality_masks[modal_name].binary_mask = np.load(os.path.join(ckpt_path, "mask_%s_bin.npy"%modal_name))
            # self.z_sizes_masked[modal_name] = int(np.sum(self.modality_masks[modal_name].binary_mask))
            self.modality_masks[modal_name].temperature = 0.2
            self.modality_masks[modal_name].trainable = False
            print(tf.sigmoid(self.modality_masks[modal_name].mask_logits))
            print(self.modality_masks[modal_name].test())
            print('Percentage of selected features for %s: '%modal_name, tf.math.reduce_sum(self.modality_masks[modal_name].binary_mask)/self.z_sizes[modal_name])
        self.shared_post_mean = np.load(os.path.join(ckpt_path,'shared_z.npy'))
        self.shared_mask.mask_logits = np.load(os.path.join(ckpt_path,'shared_mask.npy'))
        self.shared_mask.binary_mask = np.load(os.path.join(ckpt_path,'shared_mask_bin.npy'))
        self.shared_mask.temperature = 0.2
        self.shared_mask.trainable = False
        self._set_trainable_mask(trainable=False)
        self.shared_size_masked = int(np.sum(self.shared_mask.binary_mask))
        print('Percentage of selected features for shared z: ', tf.math.reduce_sum(self.shared_mask.binary_mask)/self.shared_size)
        print(tf.sigmoid(self.shared_mask.mask_logits))
        print(self.shared_mask.test())
    
    def train(self, trainloader, lr_enc=0.001, lr_dec=0.001, epochs_enc=20, epochs_dec=20):
        dec_loss, enc_loss, shared_loss, m_loss = [], [], [], []
        shared_mask = []
        modality_specific_masks = {mod: [] for mod in self.modality_names}
        n_rounds = 50
        initial_temperature = 1
        anneal_rate = -np.log(0.2 / initial_temperature) / (n_rounds-1)
        # optimizer_1 = tf.keras.optimizers.Adam(learning_rate=lr_enc, clipvalue=1)
        optimizer_2 = tf.keras.optimizers.Adam(learning_rate=lr_dec, clipvalue=1)
        optimizer_3 = tf.keras.optimizers.Adam(learning_rate=lr_dec, clipvalue=1)
        optimizer_4 = tf.keras.optimizers.Adam(learning_rate=lr_enc, clipvalue=1)
        for iter_round in range(n_rounds):
            m_loss.extend(self.optimize_modality_latent(trainloader, lr_dec, epochs_dec, optimizer_2))
            shared_loss.extend(self.optimize_shared_latent(trainloader, lr_dec, epochs_dec, optimizer_3))
            enc_loss.extend(self.train_encoder(trainloader, lr_enc, epochs_enc, optimizer_4))
            shared_mask.append(self.shared_mask.binary_mask.numpy())
            
            # Annealing temperature
            for modality_name, z_size in self.z_sizes.items():
                self.modality_masks[modality_name].temperature = tf.convert_to_tensor(initial_temperature * np.exp(-anneal_rate*(iter_round)), dtype=tf.float32)
                self.shared_mask.temperature = tf.convert_to_tensor(initial_temperature * np.exp(-anneal_rate*(iter_round)), dtype=tf.float32)
            # Track mask updates
            shared_mask.append(self.shared_mask.binary_mask.numpy())
            for mod in self.modality_names:
                modality_specific_masks[mod].append(self.modality_masks[mod].binary_mask.numpy())
            tf.keras.backend.clear_session()
            gc.collect() 

            # Prepare the data as a dictionary or list of lists
            train_tracker = {
                "modality loss": m_loss,
                "shared loss": shared_loss,
                "encoder loss": enc_loss,
                "shared mask": shared_mask,
                "modality mask": modality_specific_masks,
            }
            with open('train_tracker.json', 'w') as file:
                json.dump(train_tracker, file, indent=4)
        
        # Plot masks over iterations 
        f, axs = plt.subplots(self.n_modalitites+1, 1)
        for nn, ax in enumerate(axs):
            if nn == self.n_modalitites:
                sns.heatmap(np.array(shared_mask), ax=ax)
                ax.set_ylabel('Shared mask')
            else:
                sns.heatmap(np.array(modality_specific_masks[self.modality_names[nn]]), ax=ax)
        axs[0].set_ylabel('ECG')
        axs[1].set_ylabel('MRI')
        plt.savefig("/home/sana/multimodal/plots/masks.pdf")
        return dec_loss, enc_loss, shared_loss, m_loss


    # def train_decoder(self, trainloader, lr, n_epochs, optimizer):
    #     print(">>>>> Training the decoder ...")
    #     # loss_fn = tf.keras.losses.MeanSquaredError()
    #     self._set_trainable_mask(trainable=False)
    #     loss_trend = []   
    #     for epoch in range(n_epochs):
    #         batch_ind = 0
    #         for data,_ in trainloader:
    #             loss = 0#(self.beta*(self.shared_mask.l1())+10*(self.shared_mask.entropy_regularization())) if self.mask else 0
    #             trainable_var = []#[self.shared_mask.mask] if self.mask else []
    #             mask_var = [self.shared_mask.mask]
    #             with tf.GradientTape(persistent=True) as tape:
    #                 for m_ind, (mod, z) in enumerate(self.posterior_means.items()):
    #                     x_m = tf.convert_to_tensor(data[mod])
    #                     batch_size = x_m.shape[0]
    #                     noise = tf.random.normal(shape=(batch_size ,self.z_sizes[mod]+self.shared_size), mean=0.0, stddev=0.05)
    #                     prev_modality = self.modality_names[m_ind-1 if m_ind>0 else self.n_modalitites-1]
    #                     _, z_s_other, z_other = self.encoders[prev_modality](tf.convert_to_tensor(data[prev_modality]))
    #                     if self.mask:
    #                         z_s_other = self.shared_mask(z_s_other)
    #                         # z_s_other = self.shared_mask(z_s_other)
    #                         z_m = self.modality_masks[mod](z[batch_ind:batch_ind+batch_size])
    #                         # trainable_var.extend([self.modality_masks[mod].mask])
    #                         # loss += self.beta*(self.modality_masks[mod].l1())
    #                         # loss += 10*(self.modality_masks[mod].entropy_regularization())
    #                     else:
    #                         z_m = z[batch_ind:batch_ind+batch_size]
    #                     reconst = self.decoders[mod](tf.concat([z_m, z_s_other], -1)+noise)
    #                     loss = loss + tf.reduce_mean((x_m-reconst)**2)
    #                     trainable_var.extend(self.decoders[mod].trainable_variables)
    #                     mask_var.extend([self.modality_masks[mod].mask])
    #             gradients = tape.gradient(loss, trainable_var)
    #             optimizer.apply_gradients(zip(gradients, trainable_var))
    #             # optimizer.apply_gradients(loss, var_list=trainable_var)
    #             batch_ind += batch_size   
    #             del tape       
    #         # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.numpy()}")
    #         np.save('/home/sana/multimodal/ckpts/shared_mask', self.shared_mask.mask.numpy())
    #         np.save('/home/sana/multimodal/ckpts/shared_mask_bin', self.shared_mask.binary_mask.numpy())
    #         for (mod, decoder) in self.decoders.items():
    #             self.decoders[mod].save_weights("/home/sana/multimodal/ckpts/%s_decoder.weights.h5"%(mod))
    #             np.save('/home/sana/multimodal/ckpts/mask_%s'%mod, self.modality_masks[mod].mask.numpy())
    #             np.save('/home/sana/multimodal/ckpts/mask_%s_bin'%mod, self.modality_masks[mod].binary_mask.numpy())
    #         loss_trend.append(loss.numpy())
    #     return loss_trend
    
    def optimize_modality_latent(self, trainloader, lr, n_epochs, optimizer):
        # print(">>>>> Training the modality-specific latent ...")
        loss_fn = tf.keras.losses.MeanSquaredError()
        loss_trend = []  
        if self.mask:
            for modality_name in self.modality_names:
                self.modality_masks[modality_name].trainable = False
            self.shared_mask.trainable = True
        for epoch in range(n_epochs):
            batch_ind = 0
            for data,_ in trainloader:
                loss = (self.beta*(self.shared_mask.l1())+(10*self.shared_mask.entropy_regularization())) if self.mask else 0
                with tf.GradientTape() as tape:
                    trainable_var = [self.shared_mask.mask_logits] if self.mask else []
                    for m_ind, (mod, z) in enumerate(self.posterior_means.items()):
                        x_m = tf.convert_to_tensor(data[mod])
                        batch_size = x_m.shape[0]
                        noise = tf.random.normal(shape=(batch_size ,self.z_sizes[mod]+self.shared_size), mean=0.0, stddev=0.05)
                        prev_modality = self.modality_names[m_ind-1 if m_ind>0 else self.n_modalitites-1]
                        _, z_s_other, _ = self.encoders[prev_modality](tf.convert_to_tensor(data[prev_modality]))
                        z_s_other = self.shared_mask(z_s_other)
                        z_m = self.modality_masks[mod](z[batch_ind:batch_ind+batch_size])
                        reconst = self.decoders[mod](tf.concat([z_m, z_s_other], -1)+noise)
                        # trainable_var.extend([self.modality_masks[mod].mask])
                        # loss += self.beta*(self.modality_masks[mod].l1()) + 100*(self.modality_masks[mod].entropy_regularization())
                        # loss += loss_fn(x_m, reconst)
                        loss += tf.reduce_mean((x_m-reconst)**2)
                        loss += 0.01*tf.reduce_mean(self.posterior_means[mod][batch_ind:batch_ind+batch_size]**2)
                        trainable_var.extend([self.posterior_means[mod]])
                        trainable_var.extend(self.decoders[mod].trainable_variables)
                gradients = tape.gradient(loss, trainable_var)
                clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
                optimizer.apply_gradients(zip(gradients, trainable_var))
                batch_ind += batch_size         
                del tape   
            # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.numpy()}")
            loss_trend.append(loss.numpy())      
        if not np.isnan(loss.numpy()):
            np.save('/home/sana/multimodal/ckpts/shared_mask', self.shared_mask.mask_logits.numpy())
            np.save('/home/sana/multimodal/ckpts/shared_mask_bin', self.shared_mask.binary_mask.numpy())
            for mod, zm in self.posterior_means.items():
                self.decoders[mod].save_weights("/home/sana/multimodal/ckpts/%s_decoder.weights.h5"%(mod))
                np.save("/home/sana/multimodal/ckpts/modality_z_%s"%mod, zm.numpy())
                # np.save('/home/sana/multimodal/ckpts/mask_%s'%mod, self.modality_masks[mod].mask.numpy())
                # np.save('/home/sana/multimodal/ckpts/mask_%s_bin'%mod, self.modality_masks[mod].binary_mask.numpy())
        return loss_trend
    
    def optimize_shared_latent(self, trainloader, lr, n_epochs, optimizer):
        # print(">>>>> Training the shared latent representation ...")
        loss_fn = tf.keras.losses.MeanSquaredError()
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=1)
        loss_trend = []
        if self.mask:
            for modality_name in self.modality_names:
                self.modality_masks[modality_name].trainable = True
            self.shared_mask.trainable = False
        for epoch in range(n_epochs):
            batch_ind = 0
            for data, _ in trainloader:
                loss = 0 #(self.beta*(self.shared_mask.l1())+10*(self.shared_mask.entropy_regularization()))
                with tf.GradientTape() as tape:
                    trainable_var = [self.shared_post_mean] #+ [self.shared_mask.mask]
                    for (mod, z) in self.posterior_means.items():
                        x_m = tf.convert_to_tensor(data[mod])
                        batch_size = x_m.shape[0]
                        noise = tf.random.normal(shape=(batch_size ,self.z_sizes[mod]+self.shared_size), mean=0.0, stddev=0.05)
                        z_m = self.modality_masks[mod](z[batch_ind:batch_ind+batch_size])
                        z_s = self.shared_mask(self.shared_post_mean[batch_ind:batch_ind+batch_size])
                        reconst = self.decoders[mod](tf.concat([z_m, z_s], -1)+noise)
                        # loss += loss_fn(x_m,reconst)
                        loss += tf.reduce_mean((x_m-reconst)**2)
                        trainable_var.extend(self.decoders[mod].trainable_variables)
                        # trainable_var.extend([self.shared_mask.mask])
                        if self.mask:
                            trainable_var.extend([self.modality_masks[mod].mask_logits])
                            loss += self.beta*(self.modality_masks[mod].l1()) + (10*self.modality_masks[mod].entropy_regularization())
                    loss += 0.01*tf.reduce_mean(self.shared_post_mean[batch_ind:batch_ind+batch_size]**2)
                gradients = tape.gradient(loss, trainable_var)
                clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
                optimizer.apply_gradients(zip(gradients, trainable_var))
                batch_ind += batch_size  
                del tape          
            # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.numpy()}")
            # np.save('/home/sana/multimodal/ckpts/shared_mask', self.shared_mask.mask.numpy())
            # np.save('/home/sana/multimodal/ckpts/shared_mask_bin', self.shared_mask.binary_mask.numpy())
            loss_trend.append(loss.numpy())  
        if not np.isnan(loss.numpy()):
            # np.save('/home/sana/multimodal/ckpts/shared_mask', self.shared_mask.mask.numpy())
            # np.save('/home/sana/multimodal/ckpts/shared_mask_bin', self.shared_mask.binary_mask.numpy())
            for mod, zm in self.posterior_means.items():
                self.decoders[mod].save_weights("/home/sana/multimodal/ckpts/%s_decoder.weights.h5"%(mod))
                np.save('/home/sana/multimodal/ckpts/mask_%s'%mod, self.modality_masks[mod].mask_logits.numpy())
                np.save('/home/sana/multimodal/ckpts/mask_%s_bin'%mod, self.modality_masks[mod].binary_mask.numpy())
            np.save("/home/sana/multimodal/ckpts/shared_z", self.shared_post_mean.numpy())
        # print('Shared rep. Loss: ', np.mean(loss_trend))
        return loss_trend
    
    def train_encoder(self, trainloader, lr, n_epochs, optimizer):
        # print(">>>>> Training the encoder ...")
        # loss_fn = keras.losses.MeanSquaredError()
        # optimizer = keras.optimizers.Adam(learning_rate=lr, clipvalue=1)
        loss_trend = []
        self._set_trainable_mask(trainable=False)
        for epoch in range(n_epochs):
            batch_ind = 0
            for data,_ in trainloader:
                loss = 0
                with tf.GradientTape() as tape:
                    trainable_var = []
                    for (mod, z) in self.posterior_means.items():
                        x_m = tf.convert_to_tensor(data[mod])
                        batch_size = x_m.shape[0]
                        z_m, z_s, _ = self.encoders[mod](x_m)
                        shared_mse = (self.shared_post_mean[batch_ind:batch_ind+batch_size] - z_s)**2
                        mod_mse = (z[batch_ind:batch_ind+batch_size]-z_m)**2
                        shared_loss = tf.reduce_mean(self.shared_mask(shared_mse))
                        modality_loss = tf.reduce_mean(self.modality_masks[mod](mod_mse))
                        loss += shared_loss + modality_loss
                        trainable_var.extend(self.encoders[mod].trainable_variables)
                gradients = tape.gradient(loss, trainable_var)
                clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
                optimizer.apply_gradients(zip(gradients, trainable_var))
                del tape
                batch_ind += batch_size 
            loss_trend.append(loss.numpy())  
        if not np.isnan(loss.numpy()):
            for (mod, encoder) in self.encoders.items():
                encoder.save_weights("/home/sana/multimodal/ckpts/%s_encoder.weights.h5"%(mod))
        # print('Encoder Loss: ', np.mean(loss_trend))
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
        # z = tf.concat(z_m_all+[z_s_all[0]], axis=-1)
        z = tf.concat(z_m_all+[tf.reduce_mean(z_s_all, 0)], axis=-1)
        # z = tf.concat(z_m_all+[z_s_all[1]], axis=-1)
        self.merged_size = z.shape[-1]
        return z

    def generate_sample(self, x_batch, ind, ref_modality, target_modality, variation=False, n_samples=1, n_pca=0):
        self._set_trainable_mask(trainable=False)
        x_m = tf.convert_to_tensor(x_batch[ref_modality][ind])
        # Learn the shared space using reference modality
        z_ref, z_s, _ = self.encoders[ref_modality](tf.expand_dims(x_m,0))
        z_s = self.shared_mask(z_s)
        z_ref = self.modality_masks[ref_modality](z_ref)
        if variation:
            # Pick the top PCA samples of the missing modality (from its distribution)
            pca_m = PCA(n_components=z_sizes[target_modality], svd_solver='full')
            z_m_projected = pca_m.fit_transform(self.posterior_means[target_modality])
            max_pc_zs_inds = z_m_projected[:,n_pca].argsort()[-n_samples:]
            min_pc_zs_inds = z_m_projected[:,n_pca].argsort()[:n_samples]
            top_pos_samples_m = np.take(self.posterior_means[target_modality], max_pc_zs_inds, 0)
            top_neg_samples_m = np.take(self.posterior_means[target_modality], min_pc_zs_inds, 0)       
            modality_mask = self.modality_masks[target_modality].binary_mask
            top_pos_samples_m = self.modality_masks[target_modality](top_pos_samples_m)
            top_neg_samples_m = self.modality_masks[target_modality](top_neg_samples_m)
            z_m = tf.concat((top_pos_samples_m,top_neg_samples_m), 0)
            z_s = tf.repeat(z_s, len(z_m), 0)
            reconst_sample = self.decoders[target_modality](tf.concat([z_m, z_s], -1))
            return reconst_sample, list(max_pc_zs_inds), list(min_pc_zs_inds)
        else:
            z_ref_m = self.modality_masks[ref_modality](self.posterior_means[ref_modality])
            z_ref_s = self.shared_mask(self.shared_post_mean)
            z_ref_dist = tf.concat([z_ref_m, z_ref_s], -1)
            z_ref = tf.concat([z_ref, z_s], -1)
            # Normalize 
            z_ref = tf.nn.l2_normalize(z_ref, axis=1)  # Shape: (1, D)
            z_ref_dist = tf.nn.l2_normalize(z_ref_dist, axis=1)  # Shape: (N, D)
            # Compute cosine similarity
            cosine_similarity = tf.matmul(z_ref_dist, z_ref, transpose_b=True)
            cosine_similarity = tf.reshape(cosine_similarity, shape=(len(z_ref_dist),))  # Shape: (N,)
            # Find the top k most similar embeddings
            sample_probs = cosine_similarity/tf.reduce_sum(cosine_similarity)
            _, top_k_indices = tf.nn.top_k(cosine_similarity, k=n_samples)
            similar_samples = np.take(self.posterior_means[target_modality], top_k_indices, 0)
            similar_sample_probs = np.take(sample_probs, top_k_indices, 0)
            z_m = self.modality_masks[target_modality](similar_samples)
            z_s = tf.repeat(z_s, len(z_m), 0)
            reconst_sample = self.decoders[target_modality](tf.concat([z_m, z_s], -1))
            return reconst_sample, similar_sample_probs, x_m

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
                noise = tf.random.normal(shape=(z_m_all[mod].shape[0] ,z_m_all[mod].shape[1]+z_s_all[mod].shape[1]), mean=0.0, stddev=0.05)
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
            noise = tf.random.normal(shape=(z_m_all.shape[0] ,self.z_sizes+self.shared_size), mean=0.0, stddev=0.05)
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


class MultimodalUKBDataset(Dataset):
    def __init__(self, path, file_names, paired_rep=False):
        self.path = path
        self.file_names = file_names
        self.paired_rep = paired_rep

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
    def __init__(self, input_dim, temperature=1.0, straight_through=True, init_mask_val=0):
        super(LearnableMask, self).__init__()
        self.mask_logits = tf.Variable(initial_value=tf.zeros(input_dim)+init_mask_val, trainable=True)
        self.temperature = temperature
        self.straight_through = straight_through
        self.trainable = True
        self.binary_mask = tf.round(self.gumbel_softmax(self.mask_logits))

    def mask_size(self):
        return tf.reduce_sum(self.binary_mask)

    def l1(self):
        # mask = self.gumbel_softmax(self.mask_logits)
        mask = tf.sigmoid(self.mask_logits)
        return tf.reduce_sum(mask)

    def entropy_regularization(self):
        mask = tf.sigmoid(self.mask_logits)
        mask = tf.clip_by_value(mask, 1e-3, 1 - 1e-3) 
        entropy = -mask * tf.math.log(mask) - (1 - mask) * tf.math.log(1 - mask)
        entropy_loss = tf.reduce_sum(entropy)
        return entropy_loss
    
    def gumbel_softmax(self, logits, n_samples=0.2):
        probs = tf.sigmoid(logits)
        probs = tf.clip_by_value(probs, 1e-3, 1 - 1e-3) 
        distribution = tfp.distributions.RelaxedBernoulli(self.temperature, probs=probs, allow_nan_stats=False)
        sample = distribution.sample()
        return tf.clip_by_value(sample, 1e-12, 1.)
    
    def test(self):
        return self.gumbel_softmax(self.mask_logits)

    def call(self, x):
        # Gumbel-Softmax Sampling
        # mask = tf.nn.gumbel_softmax(self.mask, tau=self.temperature, hard=self.straight_through)
        if self.trainable is False:
            return self.binary_mask * x
        else:
            mask = self.gumbel_softmax(self.mask_logits)
            self.binary_mask = tf.round(mask)
            return x * mask  # Element-wise multiplication
 
    
