import torch
import tensorflow as tf
from tensorflow import keras 
import numpy as np
import os
from sklearn.decomposition import PCA
import gc
from utils import MultimodalUKBDataset
#for TF2
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import logistic



class DeMuReFusion():
    """
    Decoupled Multimodal representation fusion. This class integrates multiple
    modalities (e.g., images, text, audio, etc.) into a unified representation space where the 
    shared and modality-specific information is decoupled.

    Attributes:
        encoders (dict): A dictionary of pretrained unimodal encoder models, one for each modality.
                         The key is the modality name identifier.
        decoders (dict): A dictionary of decoder models (not pretrained), one for each modality.
                         The key is the modality name identifier.
        train_ids (list): A list of identifiers for training data, used for indexing or batching.
        z_sizes (dict): Size of latent representation for each modality.
        shared_size (int): Size of the shared represention.
        modality_names (list): A list of names for each modality.
        modality_shapes (dict): The input shape for samples of each modality.
        ckpt_path (str): The path to the checkpoint directory where the model's trained parameters are stored or 
                         loaded from.
        gamma (float): A hyperparameter used for entropy regularization of learnable masks.
        mask (bool): If ture, the framwork uses a trainable masks to learn the optimal size of the latent space.
        beta (float): A hyperparameter for l1 regularization of the masks.
    """
    def __init__(self, encoders, decoders, train_ids, z_sizes, shared_size,
                 modality_names, modality_shapes, ckpt_path, gamma=1, mask=False, beta=1):
        super(DeMuReFusion, self).__init__()
        self.mask = mask
        self.z_sizes = z_sizes
        self.shared_size = shared_size
        self.train_ids = [int(train_ids[idx].split('.')[0]) for idx in range(len(train_ids))]
        self.n_train = len(train_ids)
        self.n_modalitites = len(modality_names)
        self.modality_names = modality_names
        self.modality_shapes = modality_shapes
        self.beta = beta
        self.gamma = gamma
        self.ckpt_path = ckpt_path
        # Initialize latent representations
        self.posterior_means = {}
        for mod, z_size in z_sizes.items():
            self.posterior_means[mod] = tf.Variable(tf.random.normal((self.n_train,(z_size))))
        self.shared_post_mean = tf.Variable(tf.random.normal((self.n_train,(shared_size))), trainable=True)
        # Initialize masks
        self.shared_mask = LearnableMask(shared_size, init_mask_val=0, 
                                         initial_temperature=100, final_temperature=0.2)
        self.modality_masks = {}
        for modality_name, z_size in self.z_sizes.items():
            self.modality_masks[modality_name] = LearnableMask(z_size, init_mask_val=0, 
                                                               initial_temperature=100, final_temperature=0.2)
        # Initialize pretrained models
        self.encoders, self.decoders = {}, {}
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
        for layer in encoder.layers:
            layer.trainable = False
        inputs = keras.Input(shape=input_size, dtype=tf.float32)
        x = encoder(inputs, training=False)
        output1 = keras.layers.Dense(z_m_size, name='modality')(x[0])
        # fc = keras.layers.Dense(512, name='fc', activation='relu')(x[0])
        output2 = keras.layers.Dense(z_s_size, name='shared')(x[0])
        encoder = keras.models.Model(inputs=inputs, outputs=[output1, output2, x[0]])
        reps = keras.layers.Input(shape=(z_m_size+z_s_size,), dtype=tf.float32) 
        x1 = keras.layers.Dense(256, name='fusion')(reps)
        reconst = decoder(x1)
        decoder = tf.keras.models.Model(inputs=(reps), outputs=reconst[0])
        # Randomize decoder weights if the decoder is pretrained 
        for layer in decoder.layers:
            if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'kernel'):
                layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
            if hasattr(layer, 'bias_initializer') and hasattr(layer, 'bias'):
                layer.bias.assign(layer.bias_initializer(layer.bias.shape))
        for layer in decoder.layers:
            layer.trainable = True
        return encoder, decoder

    def load_from_checkpoint(self):
        ckpt_path = self.ckpt_path
        for modal_name in (self.modality_names):
            self.encoders[modal_name].load_weights(os.path.join(ckpt_path, (modal_name+"_encoder.weights.h5")))
            self.decoders[modal_name].load_weights(os.path.join(ckpt_path, (modal_name+"_decoder.weights.h5")))
            self.posterior_means[modal_name] = np.load(os.path.join(ckpt_path, "modality_z_%s.npy"%modal_name))
            self.modality_masks[modal_name].mask_logits = np.load(os.path.join(ckpt_path, "mask_%s.npy"%modal_name))
            self.modality_masks[modal_name].binary_mask = np.load(os.path.join(ckpt_path, "mask_%s_bin.npy"%modal_name))
            self.modality_masks[modal_name].temperature = 0.2
            self.modality_masks[modal_name].trainable = False
            print('Percentage of selected features for %s: '%modal_name, tf.math.reduce_sum(self.modality_masks[modal_name].binary_mask)/self.z_sizes[modal_name])
            print(tf.sigmoid(self.modality_masks[modal_name].mask_logits))
        self.shared_post_mean = np.load(os.path.join(ckpt_path,'shared_z.npy'))
        self.shared_mask.mask_logits = np.load(os.path.join(ckpt_path,'shared_mask.npy'))
        self.shared_mask.binary_mask = np.load(os.path.join(ckpt_path,'shared_mask_bin.npy'))
        self.shared_mask.temperature = 0.2
        self.shared_mask.trainable = False
        self._set_trainable_mask(trainable=False)
        self.shared_size_masked = int(np.sum(self.shared_mask.binary_mask))
        print('Percentage of selected features for shared z: ', tf.math.reduce_sum(self.shared_mask.binary_mask)/self.shared_size)
        print(np.array(tf.sigmoid(self.shared_mask.mask_logits)).tolist())
    
    def train(self, trainloader, lr_enc=0.001, lr_dec=0.001, epochs_enc=20, epochs_dec=20, iteration_count=30, temp_annealing='exponential', no_mask_epochs=None, extra_encoder_training=None):
        dec_loss, enc_loss, shared_loss, m_loss = [], [], [], []
        shared_mask = []
        modality_specific_masks = {mod: [] for mod in self.modality_names}
        optimizer_1 = tf.keras.optimizers.Adam(learning_rate=lr_dec, clipvalue=1)
        optimizer_2 = tf.keras.optimizers.Adam(learning_rate=lr_dec, clipvalue=1)
        optimizer_3 = tf.keras.optimizers.Adam(learning_rate=lr_dec, clipvalue=1)
        optimizer_4 = tf.keras.optimizers.Adam(learning_rate=lr_dec, clipvalue=1)
        optimizer_encoder_1 = tf.keras.optimizers.Adam(learning_rate=lr_enc, clipvalue=1)
        optimizer_encoder_2 = tf.keras.optimizers.Adam(learning_rate=lr_enc, clipvalue=1)

        # Run the first round without the masking
        if no_mask_epochs is not None and self.mask is True:
            self.mask = False
            m_loss.extend(self.optimize_modality_latent(trainloader, lr_dec, no_mask_epochs, [optimizer_1,optimizer_2]))
            shared_loss.extend(self.optimize_shared_latent(trainloader, lr_dec, no_mask_epochs, [optimizer_3,optimizer_4]))
            enc_loss.extend(self.train_encoder(trainloader, lr_enc, no_mask_epochs, [optimizer_encoder_1, optimizer_encoder_2]))
            optimizer_1 = tf.keras.optimizers.Adam(learning_rate=lr_dec, clipvalue=1)
            optimizer_2 = tf.keras.optimizers.Adam(learning_rate=lr_dec, clipvalue=1)
            optimizer_3 = tf.keras.optimizers.Adam(learning_rate=lr_dec, clipvalue=1)
            optimizer_4 = tf.keras.optimizers.Adam(learning_rate=lr_dec, clipvalue=1)
            self.mask = True

        for iter_round in range(iteration_count):
            m_loss.extend(self.optimize_modality_latent(trainloader, lr_dec, epochs_dec, [optimizer_1,optimizer_2]))
            shared_loss.extend(self.optimize_shared_latent(trainloader, lr_dec, epochs_dec, [optimizer_3,optimizer_4]))
            enc_loss.extend(self.train_encoder(trainloader, lr_enc, epochs_enc, [optimizer_encoder_1, optimizer_encoder_2]))
            # Annealing and recording temperature
            for modality_name, z_size in self.z_sizes.items():
                self._anneal_temperature(modality_name=modality_name, annealing_strategy=temp_annealing, iter_round=iter_round, n_rounds=iteration_count)
                modality_specific_masks[modality_name].append(self.modality_masks[modality_name].binary_mask.numpy())
            self._anneal_temperature(modality_name='shared', annealing_strategy=temp_annealing, iter_round=iter_round, n_rounds=iteration_count)           
            shared_mask.append(self.shared_mask.binary_mask.numpy()) 
            tf.keras.backend.clear_session()
            gc.collect() 
            # Prepare the data as a dictionary or list of lists
            np.savez(os.path.join(self.ckpt_path,'train_tracker.npz'), modality_loss=m_loss, shared_loss=shared_loss, 
                    encoder_loss=enc_loss, shared_mask=np.stack(shared_mask),
                    m1_mask= np.stack(modality_specific_masks[self.modality_names[0]]),
                    m2_mask= np.stack(modality_specific_masks[self.modality_names[1]]))

        if extra_encoder_training is not None:
            enc_loss.extend(self.train_encoder(trainloader, lr_enc, extra_encoder_training, [optimizer_encoder_1, optimizer_encoder_2]))
            np.savez(os.path.join(self.ckpt_path,'train_tracker.npz'), modality_loss=m_loss, shared_loss=shared_loss, 
                        encoder_loss=enc_loss, shared_mask=np.stack(shared_mask),
                        m1_mask= np.stack(modality_specific_masks[self.modality_names[0]]),
                        m2_mask= np.stack(modality_specific_masks[self.modality_names[1]])) 
        return dec_loss, enc_loss, shared_loss, m_loss
    
    def _anneal_temperature(self, modality_name, annealing_strategy, iter_round, n_rounds):
        if modality_name=='shared':
            if annealing_strategy=='exponential':
                anneal_rate_shared = -np.log(self.shared_mask.final_temperature / self.shared_mask.initial_temperature) / (n_rounds-1)
                self.shared_mask.temperature = tf.convert_to_tensor(self.shared_mask.initial_temperature * np.exp(-anneal_rate_shared*(iter_round)), dtype=tf.float32) 
            elif annealing_strategy=='linear':
                anneal_rate_shared = (self.shared_mask.final_temperature - self.shared_mask.initial_temperature) / (n_rounds-1)
                self.shared_mask.temperature = self.shared_mask.initial_temperature - anneal_rate_shared*iter_round       
        else:
            initial_temperature = self.modality_masks[modality_name].initial_temperature
            final_temperature = self.modality_masks[modality_name].final_temperature
            if annealing_strategy=='exponential':
                anneal_rate = -np.log(final_temperature / initial_temperature) / (n_rounds-1)
                self.modality_masks[modality_name].temperature = tf.convert_to_tensor(initial_temperature * np.exp(-anneal_rate*(iter_round)), dtype=tf.float32)
            elif annealing_strategy=='linear':
                anneal_rate = (final_temperature - initial_temperature) / (n_rounds-1)
                self.modality_masks[modality_name].temperature = initial_temperature - anneal_rate*iter_round

    def optimize_modality_latent(self, trainloader, lr, n_epochs, optimizers):
        loss_fn = tf.keras.losses.MeanSquaredError()
        loss_trend = []  
        if self.mask:
            for modality_name in self.modality_names:
                self.modality_masks[modality_name].trainable = False
            self.shared_mask.trainable = True
        
        for epoch in range(n_epochs):
            batch_ind = 0
            epoch_loss = []
            for data,_ in trainloader:
                batch_loss = 0
                for m_ind, (mod, z) in enumerate(self.posterior_means.items()):
                    with tf.GradientTape() as tape:
                        loss = 0
                        trainable_var = [self.shared_mask.mask_logits] if (self.mask) else []
                        x_m = tf.convert_to_tensor(data[mod])
                        batch_size = x_m.shape[0]
                        noise_m = tf.random.normal(shape=(batch_size ,self.z_sizes[mod]), mean=0.0, stddev=0.1)
                        noise_s = tf.random.normal(shape=(batch_size ,self.shared_size), mean=0.0, stddev=0.1)
                        prev_modality = self.modality_names[m_ind-1 if m_ind>0 else self.n_modalitites-1]
                        _, z_s_other, _ = self.encoders[prev_modality](tf.convert_to_tensor(data[prev_modality]))
                        z_s_other = z_s_other+noise_s
                        z_m = z[batch_ind:batch_ind+batch_size]+noise_m
                        mse = self.posterior_means[mod][batch_ind:batch_ind+batch_size]**2
                        if self.mask:
                            loss += (0.1*self.beta*(self.shared_mask.l1())+self.gamma*(self.shared_mask.entropy_regularization())) 
                            z_s_other = self.shared_mask(z_s_other)
                            z_m = self.modality_masks[mod](z_m)
                        reconst = self.decoders[mod](tf.concat([z_m, z_s_other], -1))
                        loss += loss_fn(x_m, reconst)
                        loss += 0.01*tf.reduce_mean(mse)
                        trainable_var.extend([self.posterior_means[mod]])
                        trainable_var.extend(self.decoders[mod].trainable_variables)
                    gradients = tape.gradient(loss, trainable_var)
                    optimizers[m_ind].apply_gradients(zip(gradients, trainable_var))
                    batch_loss += loss.numpy()
                    del tape
                batch_ind += batch_size     
                epoch_loss.append(batch_loss) 
            loss_trend.append(np.mean(epoch_loss))      
        if not np.isnan(loss.numpy()):
            np.save(os.path.join(self.ckpt_path,'shared_mask'), self.shared_mask.mask_logits.numpy())
            np.save(os.path.join(self.ckpt_path,'shared_mask_bin'), self.shared_mask.binary_mask.numpy())
            for mod, zm in self.posterior_means.items():
                self.decoders[mod].save_weights(os.path.join(self.ckpt_path,"%s_decoder.weights.h5"%(mod)))
                np.save(os.path.join(self.ckpt_path,"modality_z_%s"%mod), zm.numpy())
        return loss_trend
    
    def optimize_shared_latent(self, trainloader, lr, n_epochs, optimizers):
        loss_fn = tf.keras.losses.MeanSquaredError()
        loss_trend = []
        if self.mask:
            for modality_name in self.modality_names:
                self.modality_masks[modality_name].trainable = True
            self.shared_mask.trainable = False
        for epoch in range(n_epochs):
            batch_ind = 0
            epoch_loss = []
            for data,_ in trainloader:
                batch_loss = 0
                for m_ind, (mod, z) in enumerate(self.posterior_means.items()):
                    with tf.GradientTape() as tape:
                        trainable_var = [self.shared_post_mean]
                        x_m = tf.convert_to_tensor(data[mod])
                        batch_size = x_m.shape[0]
                        mse = self.shared_post_mean[batch_ind:batch_ind+batch_size]**2           
                        noise_m = tf.random.normal(shape=(batch_size ,self.z_sizes[mod]), mean=0.0, stddev=0.1)
                        noise_s = tf.random.normal(shape=(batch_size ,self.shared_size), mean=0.0, stddev=0.1)
                        z_m = z[batch_ind:batch_ind+batch_size]+noise_m
                        z_s = self.shared_post_mean[batch_ind:batch_ind+batch_size]+noise_s
                        if self.mask:
                            z_m = self.modality_masks[mod](z_m)
                            z_s = self.shared_mask(z_s)
                        loss = 0.01*tf.reduce_mean(mse)
                        reconst = self.decoders[mod](tf.concat([z_m, z_s], -1))
                        loss += loss_fn(x_m,reconst)
                        # loss += 0.01*self._frobenius_distance_cosine(z_m, z_s)
                        trainable_var.extend(self.decoders[mod].trainable_variables)
                        if self.mask:
                            trainable_var.extend([self.modality_masks[mod].mask_logits])
                            loss += self.beta*(self.modality_masks[mod].l1()) + (self.gamma*self.modality_masks[mod].entropy_regularization())
                    gradients = tape.gradient(loss, trainable_var)
                    optimizers[m_ind].apply_gradients(zip(gradients, trainable_var))
                    batch_loss += loss.numpy()
                    del tape 
                batch_ind += batch_size 
                epoch_loss.append(batch_loss) 
            loss_trend.append(np.mean(epoch_loss))  
        if not np.isnan(loss.numpy()):
            for mod, zm in self.posterior_means.items():
                self.decoders[mod].save_weights(os.path.join(self.ckpt_path,"/%s_decoder.weights.h5"%(mod)))
                np.save(os.path.join(self.ckpt_path,'mask_%s'%mod), self.modality_masks[mod].mask_logits.numpy())
                np.save(os.path.join(self.ckpt_path,'mask_%s_bin'%mod), self.modality_masks[mod].binary_mask.numpy())
            np.save(os.path.join(self.ckpt_path,"shared_z"), self.shared_post_mean.numpy())
        return loss_trend
    
    def train_encoder(self, trainloader, lr, n_epochs, optimizers):
        loss_trend = []
        self._set_trainable_mask(trainable=False)
        for epoch in range(n_epochs):
            batch_ind = 0
            epoch_loss = []
            for data,_ in trainloader:
                batch_loss = 0
                for m_ind, (mod, encoder) in  enumerate(self.encoders.items()):
                    with tf.GradientTape() as tape:
                        x_m = tf.convert_to_tensor(data[mod])
                        batch_size = x_m.shape[0]
                        z_m, z_s, _ = encoder(x_m)
                        shared_mse = (self.shared_post_mean[batch_ind:batch_ind+batch_size] - z_s)**2
                        mod_mse = (self.posterior_means[mod][batch_ind:batch_ind+batch_size]-z_m)**2
                        shared_loss = tf.reduce_sum(self.shared_mask(shared_mse), -1)/tf.reduce_sum(self.shared_mask.binary_mask)
                        modality_loss = tf.reduce_sum(self.modality_masks[mod](mod_mse), -1)/tf.reduce_sum(self.modality_masks[mod].binary_mask)
                        loss = 10*tf.reduce_mean(shared_loss) + tf.reduce_mean(modality_loss)
                        trainable_var = encoder.trainable_variables
                    gradients = tape.gradient(loss, trainable_var)
                    optimizers[m_ind].apply_gradients(zip(gradients, trainable_var))
                    del tape
                batch_loss += loss.numpy()
                batch_ind += batch_size 
                epoch_loss.append(batch_loss)
            loss_trend.append(np.mean(epoch_loss))  
        if not np.isnan(loss.numpy()):
            for (mod, encoder) in self.encoders.items():
                encoder.save_weights(os.path.join(self.ckpt_path,"%s_encoder.weights.h5"%(mod)))
        return loss_trend
    
    def merge_representations(self, x_batch):
        self._set_trainable_mask(trainable=False)
        z_m_all, z_s_all = [], []
        for mod, encoder in (self.encoders).items():
            z_m, z_s, _ = encoder(tf.convert_to_tensor(x_batch[mod]))
            modality_mask = self.modality_masks[mod].binary_mask
            z_m = np.take(z_m, np.argwhere(modality_mask==1)[:,0], axis=1)
            z_m_all.append(z_m)
            shared_mask = self.shared_mask.binary_mask
            z_s = np.take(z_s, np.argwhere(shared_mask==1)[:,0], axis=1)
            z_s_all.append(z_s)
        z = tf.concat(z_m_all+[tf.reduce_mean(z_s_all, 0)], axis=-1)
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
            modality_mask = self.modality_masks[target_modality].binary_mask
            pca_m = PCA(n_components=int(np.sum(modality_mask)), svd_solver='full')
            low_dim_data = np.take(self.posterior_means[target_modality], np.argwhere(modality_mask==1)[:,0], axis=1)
            z_m_projected = pca_m.fit_transform(low_dim_data)
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
            # Generate most likely pairs for a reference sample
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
                    modality_mask = self.modality_masks[mod].binary_mask
                    z_m_all[mod] = np.take(z_m, np.argwhere(modality_mask==1)[:,0], axis=1)
                    shared_mask = self.shared_mask.binary_mask
                    z_s_all[mod] = np.take(z_s, np.argwhere(shared_mask==1)[:,0], axis=1)
                else:
                    z_s_all[mod] = self.shared_mask(z_s)
                    z_m_all[mod] = self.modality_masks[mod](z_m)
            else:
                z_s_all[mod] = z_s
                z_m_all[mod] = z_m
        return z_s_all, z_m_all

    def decode(self, z_s_all=None, z_m_all=None):
        self._set_trainable_mask(trainable=False)
        x_recon = {}
        if z_s_all is None and z_m_all is None: # reconstruct an image from the posterior latent
            for m_ind, (mod, decoder) in enumerate((self.decoders).items()):
                if self.mask:
                    x_recon[mod] = decoder(tf.concat([self.modality_masks[mod](self.posterior_means[mod][:5]), 
                                                          self.shared_mask(self.shared_post_mean[:5])], -1))
                else:
                    x_recon[mod] = decoder(tf.concat([self.posterior_means[mod][:5], self.shared_post_mean[:5]], -1))
        else:
            for m_ind, (mod, decoder) in enumerate((self.decoders).items()):
                if self.mask:
                    x_recon[mod] = decoder(tf.concat([self.modality_masks[mod](z_m_all[mod]), 
                                                        self.shared_mask(z_s_all[mod])], -1))
                else:
                    x_recon[mod] = decoder(tf.concat([z_m_all[mod], z_s_all[mod]], -1))
        return x_recon

    def _frobenius_distance_cosine(self, X, Y):
        # Normalize the samples to ensure vectors have unit length
        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y_normalized = Y / np.linalg.norm(Y, axis=1, keepdims=True)

        # Compute the cosine similarity matrix between X and Y
        cosine_similarity_matrix = np.dot(X_normalized, Y_normalized.T)

        # Compute the Frobenius norm of the cosine similarity matrix
        frobenius_norm = np.linalg.norm(cosine_similarity_matrix, ord='fro')

        return frobenius_norm


class LearnableMask(keras.layers.Layer):
    def __init__(self, input_dim, straight_through=True, init_mask_val=0, initial_temperature=1, final_temperature=0.2):
        super(LearnableMask, self).__init__()
        self.initial_temperature = initial_temperature
        self.mask_logits = tf.Variable(initial_value=tf.zeros(input_dim)+init_mask_val, trainable=True)
        self.temperature = initial_temperature
        self.straight_through = straight_through
        self.trainable = True
        self.final_temperature = final_temperature
        self.binary_mask = tf.round(self.gumbel_softmax(self.mask_logits))

    def mask_size(self):
        return tf.reduce_sum(self.binary_mask)

    def l1(self):
        mask = tf.sigmoid(self.mask_logits)
        return tf.reduce_sum(mask)

    def entropy(self):
        probabilities = tf.sigmoid(self.mask_logits)
        entropy = -tf.reduce_mean(probabilities * tf.math.log(probabilities + 1e-9)+(1-probabilities) * tf.math.log((1-probabilities) + 1e-9))
        return entropy

    def entropy_regularization(self):
        mask = tf.sigmoid(self.mask_logits)
        mask = tf.clip_by_value(mask, 1e-3, 1 - 1e-3) 
        entropy = -mask * tf.math.log(mask) - (1 - mask) * tf.math.log(1 - mask)
        entropy_loss = tf.reduce_mean(entropy)
        return entropy_loss
    
    def gumbel_softmax(self, logits, n_samples=0.2):
        probs = tf.sigmoid(logits)
        probs = tf.clip_by_value(probs, 1e-2, 1 - 1e-2) 
        distribution = tfp.distributions.RelaxedBernoulli(self.temperature, probs=probs, allow_nan_stats=False)
        sample = distribution.sample()
        return tf.clip_by_value(sample, 1e-6, 1.)
    
    def test(self):
        return self.gumbel_softmax(self.mask_logits)

    def call(self, x):
        if self.trainable is False:
            return self.binary_mask * x
        else:
            mask = self.gumbel_softmax(self.mask_logits)
            self.binary_mask = tf.round(mask)
            return x * mask  # Element-wise multiplication
 
    




