import torch
from torch.utils.data import Dataset
import tensorflow as tf
from tensorflow import keras 
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
# from tf_agents.distributions.gumbel_softmax import GumbelSoftmax 
sns.set_theme()
# from keras.utils.layer_utils import count_params
from ml4h.tensormap.ukb.ecg import ecg_rest_median_raw_10
from ml4h.tensormap.ukb.mri import lax_4ch_heart_center
#for TF2
import tensorflow_probability as tfp



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
        
        self.posterior_means = {}
        for mod, z_size in z_sizes.items():
            self.posterior_means[mod] = tf.Variable(tf.random.normal((self.n_train,(z_size))))
        # self.posterior_means = [tf.Variable(tf.random.normal((self.n_train,(z_size))), trainable=True) for z_size in z_sizes.values()]
        self.shared_post_mean = tf.Variable(tf.random.normal((self.n_train,(shared_size))), trainable=True)
        self.shared_mask = LearnableMask(shared_size)
        self.modality_masks = {}
        for modality_name, z_size in self.z_sizes.items():
            self.modality_masks[modality_name] = LearnableMask(z_size)
        
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
        if not trainable:
            for k,v in encoder._get_trainable_state().items():
                k.trainable = False
            for k,v in decoder._get_trainable_state().items():
                k.trainable = False
        inputs = keras.Input(shape=input_size, dtype=tf.float32)
        x = encoder(inputs, training=False)
        output1 = keras.layers.Dense(z_m_size, name='modality')(x[0])
        output2 = keras.layers.Dense(z_s_size, name='shared')(x[0])
        # encoder = keras.Model(inputs=inputs, outputs=[output1, output2, x])
        encoder = keras.models.Model(inputs=inputs, outputs=[output1, output2, x[0]])
        # reps = keras.Input(shape=(z_m_size+z_s_size)) 
        reps = keras.layers.Input(shape=(z_m_size+z_s_size,), dtype=tf.float32) 
        x1 = keras.layers.Dense(256, name='fusion')(reps)
        reconst = decoder(x1)
        # decoder = tf.keras.Model(inputs=(reps), outputs=reconst)
        decoder = tf.keras.models.Model(inputs=(reps), outputs=reconst[0])
        return encoder, decoder

    def load_from_checkpoint(self, ckpt_path):
        for modal_name in (self.modality_names):
            self.encoders[modal_name].load_weights(os.path.join(ckpt_path, (modal_name+"_encoder.weights.h5")))
            self.decoders[modal_name].load_weights(os.path.join(ckpt_path, (modal_name+"_decoder.weights.h5")))
            self.posterior_means[modal_name] = np.load(os.path.join(ckpt_path, "modality_z_%s.npy"%modal_name))
            self.modality_masks[modal_name].mask = np.load(os.path.join(ckpt_path, "mask_%s.npy"%modal_name))
            self.modality_masks[modal_name].binary_mask = np.load(os.path.join(ckpt_path, "mask_%s_bin.npy"%modal_name))
            print('Percentage of selected features for %s: '%modal_name, tf.math.reduce_sum(self.modality_masks[modal_name].binary_mask)/self.z_sizes[modal_name])
        self.shared_post_mean = np.load('/home/sana/multimodal/ckpts/shared_z.npy')
        self.shared_mask.mask = np.load('/home/sana/multimodal/ckpts/shared_mask.npy')
        self.shared_mask.binary_mask = np.load('/home/sana/multimodal/ckpts/shared_mask_bin.npy')
        self._set_trainable_mask(trainable=False)
        print('Percentage of selected features for shared z: ', tf.math.reduce_sum(self.shared_mask.binary_mask)/self.shared_size)
        print(self.shared_mask.mask)
    
    def train(self, trainloader, lr_enc=0.001, lr_dec=0.001, epochs_enc=20, epochs_dec=20):
        dec_loss, enc_loss, shared_loss, m_loss = [], [], [], []
        shared_mask = []
        modality_specific_masks = {mod: [] for mod in self.modality_names}
        for iter_round in range(15):
            dec_loss.extend(self.train_decoder(trainloader, lr_dec, epochs_dec))
            m_loss.extend(self.optimize_modality_latent(trainloader, lr_dec, epochs_dec))
            shared_loss.extend(self.optimize_shared_latent(trainloader, lr_dec, epochs_dec))
            enc_loss.extend(self.train_encoder(trainloader, lr_enc, epochs_enc))
            # Update temperature
            for modality_name, z_size in self.z_sizes.items():
                self.modality_masks[modality_name].temperature = self.modality_masks[modality_name].temperature - 0.04
            self.shared_mask.temperature = self.shared_mask.temperature - 0.04
            print(self.shared_mask.temperature, self.modality_masks[modality_name].temperature)
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

    def train_decoder(self, trainloader, lr, n_epochs):
        print(">>>>> Training the decoder ...")
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1)
        self._set_trainable_mask(trainable=True)
        loss_trend = []   
        for epoch in range(n_epochs):
            batch_ind = 0
            for data,_ in trainloader:
                loss = self.beta*tf.reduce_sum(self.shared_mask.l1()) if self.mask else 0
                trainable_var = [self.shared_mask.mask] if self.mask else []
                with tf.GradientTape() as tape:
                    for m_ind, (mod, z) in enumerate(self.posterior_means.items()):
                        x_m = tf.convert_to_tensor(data[mod])
                        batch_size = x_m.shape[0]
                        noise = tf.random.normal(shape=(batch_size ,self.z_sizes[mod]), mean=0.0, stddev=0.2)
                        prev_modality = self.modality_names[m_ind-1 if m_ind>0 else self.n_modalitites-1]
                        _,conditional_input,_ = self.encoders[prev_modality](tf.convert_to_tensor(data[prev_modality]))
                        if self.mask:
                            conditional_input = self.shared_mask(conditional_input)
                            z_m = self.modality_masks[mod](z[batch_ind:batch_ind+batch_size])
                            trainable_var.extend([self.modality_masks[mod].mask])
                            loss += self.beta*tf.reduce_sum(self.modality_masks[mod].l1())
                        else:
                            z_m = z[batch_ind:batch_ind+batch_size]
                        reconst = self.decoders[mod](tf.concat([z_m+noise, conditional_input], -1))
                        loss = loss + loss_fn(x_m, reconst)
                        trainable_var.extend(self.decoders[mod].trainable_variables)
                    gradients = tape.gradient(loss, trainable_var)
                    optimizer.apply_gradients(zip(gradients, trainable_var))
                    batch_ind += batch_size            
            # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.numpy()}")
            np.save('/home/sana/multimodal/ckpts/shared_mask', self.shared_mask.mask.numpy())
            np.save('/home/sana/multimodal/ckpts/shared_mask_bin', self.shared_mask.binary_mask.numpy())
            for (mod, decoder) in self.decoders.items():
                decoder.save_weights("/home/sana/multimodal/ckpts/%s_decoder.weights.h5"%(mod))
                np.save('/home/sana/multimodal/ckpts/mask_%s'%mod, self.modality_masks[mod].mask.numpy())
                np.save('/home/sana/multimodal/ckpts/mask_%s_bin'%mod, self.modality_masks[mod].binary_mask.numpy())
            loss_trend.append(loss.numpy())
        print('Decoder Loss: ', np.mean(loss_trend))
        return loss_trend
    
    def optimize_modality_latent(self, trainloader, lr, n_epochs):
        print(">>>>> Training the modality-specific latent ...")
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1)
        loss_trend = []  
        self._set_trainable_mask(trainable=False)
        for epoch in range(n_epochs):
            batch_ind = 0
            for data,_ in trainloader:
                loss = 0
                trainable_var = []
                with tf.GradientTape() as tape:
                    for m_ind, (mod, z) in enumerate(self.posterior_means.items()):
                        x_m = tf.convert_to_tensor(data[mod])
                        batch_size = x_m.shape[0]
                        noise = tf.random.normal(shape=(batch_size ,self.z_sizes[mod]), mean=0.0, stddev=0.2)
                        prev_modality = self.modality_names[m_ind-1 if m_ind>0 else self.n_modalitites-1]
                        _,conditional_input,_ = self.encoders[prev_modality](tf.convert_to_tensor(data[prev_modality]))
                        if self.mask:
                            conditional_input = self.shared_mask(conditional_input)
                            z_m = self.modality_masks[mod](z[batch_ind:batch_ind+batch_size])
                        else:
                            z_m = z[batch_ind:batch_ind+batch_size]
                        reconst = self.decoders[mod](tf.concat([z_m+noise, conditional_input], -1))
                        loss += loss_fn(x_m, reconst)
                        loss += tf.reduce_mean(tf.reduce_sum(self.posterior_means[mod][batch_ind:batch_ind+batch_size]**2, axis=-1))
                        trainable_var.extend([self.posterior_means[mod]])
                    gradients = tape.gradient(loss, trainable_var)
                    optimizer.apply_gradients(zip(gradients, trainable_var))
                    batch_ind += batch_size            
            # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.numpy()}")
            loss_trend.append(loss.numpy())      
            for mod, zm in self.posterior_means.items():
                np.save("/home/sana/multimodal/ckpts/modality_z_%s"%mod, zm.numpy())
        return loss_trend
    
    def optimize_shared_latent(self, trainloader, lr, n_epochs):
        print(">>>>> Training the shared latent representation ...")
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1)
        loss_trend = []
        self._set_trainable_mask(trainable=False)
        for epoch in range(n_epochs):
            batch_ind = 0
            for data,_ in trainloader:
                loss = 0
                trainable_var = [self.shared_post_mean]
                with tf.GradientTape() as tape:
                    for (mod, z) in self.posterior_means.items():
                        x_m = tf.convert_to_tensor(data[mod])
                        batch_size = x_m.shape[0]
                        noise = tf.random.normal(shape=(batch_size ,self.z_sizes[mod]+self.shared_size), mean=0.0, stddev=0.5)
                        if self.mask:
                            z_m = self.modality_masks[mod](z[batch_ind:batch_ind+batch_size])
                            z_s = self.shared_mask(self.shared_post_mean[batch_ind:batch_ind+batch_size])
                        else:
                            z_m = z[batch_ind:batch_ind+batch_size]
                            z_s = self.shared_post_mean[batch_ind:batch_ind+batch_size]
                        reconst = self.decoders[mod](tf.concat([z_m, z_s], -1)+noise)
                        loss += loss_fn(x_m, reconst)
                        loss += tf.reduce_mean(tf.reduce_sum(self.shared_post_mean[batch_ind:batch_ind+batch_size]**2, -1))
                    gradients = tape.gradient(loss, trainable_var)
                    optimizer.apply_gradients(zip(gradients, trainable_var))
                    batch_ind += batch_size            
            # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.numpy()}")
            values = self.shared_post_mean.numpy()
            np.save("/home/sana/multimodal/ckpts/shared_z", values)
            loss_trend.append(loss.numpy())
        return loss_trend
    
    def train_encoder(self, trainloader, lr, n_epochs):
        print(">>>>> Training the encoder ...")
        loss_fn = keras.losses.MeanSquaredError()
        optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1)
        loss_trend = []
        self._set_trainable_mask(trainable=True)
        for epoch in range(n_epochs):
            batch_ind = 0
            for data,_ in trainloader:
                loss = self.beta*tf.reduce_sum(self.shared_mask.l1()) if self.mask else 0
                trainable_var = [self.shared_mask.mask] if self.mask else []
                with tf.GradientTape() as tape:
                    for (mod, z) in self.posterior_means.items():
                        x_m = tf.convert_to_tensor(data[mod])
                        batch_size = x_m.shape[0]
                        z_m, z_s, _ = self.encoders[mod](x_m)
                        if self.mask:
                            # shared_loss = tf.math.reduce_sum(self.shared_mask(self.shared_post_mean[batch_ind:batch_ind+batch_size] - z_s)**2)
                            # modality_loss = tf.math.reduce_sum(self.modality_masks[mod](z[batch_ind:batch_ind+batch_size] - z_m)**2)
                            shared_loss = loss_fn(self.shared_mask(self.shared_post_mean[batch_ind:batch_ind+batch_size]), z_s)
                            modality_loss = loss_fn(self.modality_masks[mod](z[batch_ind:batch_ind+batch_size]), z_m)
                            trainable_var.extend([self.modality_masks[mod].mask])
                            loss += self.beta*tf.reduce_sum(self.modality_masks[mod].l1())
                        else:
                            shared_loss = tf.reduce_sum(self.shared_mask((self.shared_post_mean[batch_ind:batch_ind+batch_size] - z_s))**2)#/max(self.shared_mask.mask_size(), 1)
                            modality_loss = tf.reduce_sum(self.modality_masks[mod]((z[batch_ind:batch_ind+batch_size]-z_m))**2)#/max(self.modality_masks[mod].mask_size(), 1)
                        loss += shared_loss + modality_loss
                        trainable_var.extend(self.encoders[mod].trainable_variables)
                    gradients = tape.gradient(loss, trainable_var)
                    optimizer.apply_gradients(zip(gradients, trainable_var))
                    batch_ind += batch_size 
            loss_trend.append(loss.numpy())
            # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.numpy()}")
            np.save('/home/sana/multimodal/ckpts/shared_mask', self.shared_mask.mask.numpy())
            np.save('/home/sana/multimodal/ckpts/shared_mask_bin', self.shared_mask.binary_mask.numpy())
            for (mod, encoder) in self.encoders.items():
                encoder.save_weights("/home/sana/multimodal/ckpts/%s_encoder.weights.h5"%(mod))
                np.save('/home/sana/multimodal/ckpts/mask_%s'%mod, self.modality_masks[mod].mask.numpy())
                np.save('/home/sana/multimodal/ckpts/mask_%s_bin'%mod, self.modality_masks[mod].binary_mask.numpy())
        print('Encoder Loss: ', np.mean(loss_trend))
        return loss_trend
    
    def merge_representations(self, x_batch):
        self._set_trainable_mask(trainable=False)
        z_m_all, z_s_all = [], []
        for mod, encoder in (self.encoders).items():
            x_m = tf.convert_to_tensor(x_batch[mod])
            z_m, z_s, _ = encoder(x_m)
            modality_mask = self.modality_masks[mod].binary_mask
            z_m = np.take(z_m, np.argwhere(modality_mask==1)[:,0], axis=1)
            z_m_all.append(z_m)
            shared_mask = self.shared_mask.binary_mask
            z_s = np.take(z_s, np.argwhere(shared_mask==1)[:,0], axis=1)
            if z_s_all==0:
                z_s_all = z_s
            else:
                z_s_all += z_s
        z = tf.concat(z_m_all+[z_s_all/self.n_modalitites], axis=-1)
        print(z.shape)
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


class MultimodalDataset(Dataset):
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
    def __init__(self, input_dim, temperature=1.0, straight_through=True):
        super(LearnableMask, self).__init__()
        self.mask = tf.Variable(initial_value=tf.ones(input_dim), trainable=True)
        self.temperature = temperature
        self.straight_through = straight_through
        self.trainable = True
        self.binary_mask = tf.Variable(initial_value=tf.ones(input_dim), trainable=False)

    def mask_size(self):
        return tf.reduce_sum(self.binary_mask)

    def l1(self):
        mask = self.gumbel_softmax(self.mask)
        return tf.reduce_sum(mask)
    
    def gumbel_softmax(self, logits, n_samples=1):
        probs = tf.sigmoid(logits)
        probs = tf.clip_by_value(probs, 1e-3, 1 - 1e-3) 
        distribution = tfp.distributions.RelaxedBernoulli(self.temperature, probs=probs, allow_nan_stats=False)
        return distribution.sample()
    
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
    
    def get_binary_mask(self):
        bin_mask = tf.reduce_mean(self.gumbel_softmax(self.mask, n_samples=1), 0)
        bin_mask = tf.round(bin_mask)
        return bin_mask
    
