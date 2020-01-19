"""
This code is adapted from https://github.com/pamattei/miwae
For more information on MIWAE see:
http://proceedings.mlr.press/v97/mattei19a.html
"""
import tensorflow as tf
import numpy as np

import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers
from utils import mse
from utils import load_data


def main(p_miss=0.5, hidden_units=50, lr=0.001, epochs=500, dataset="drive",
         mode="mcar", para=0.5, train=None, rand_seed=42):

    np.random.seed(rand_seed)
    tf.set_random_seed(rand_seed)

    n, p, xmiss, xhat_0, mask, data_x, data_y = load_data(p_miss, dataset=dataset, mode=mode, para=para, train=train, rand_seed=rand_seed)

    x = tf.placeholder(tf.float32, shape=[None, p]) # Placeholder for xhat_0
    learning_rate = tf.placeholder(tf.float32, shape=[])
    batch_size = tf.shape(x)[0]
    xmask = tf.placeholder(tf.bool, shape=[None, p])
    K= tf.placeholder(tf.int32, shape=[]) # Placeholder for the number of importance weights

    d = np.floor(p/2).astype(int) # dimension of the latent space

    p_z = tfd.MultivariateNormalDiag(loc=tf.zeros(d, tf.float32))

    h = hidden_units # number of hidden units (same for all MLPs)

    sigma = "relu"

    decoder = tfk.Sequential([
      tfkl.InputLayer(input_shape=[d,]),
      tfkl.Dense(h, activation=sigma,kernel_initializer="orthogonal"),
      tfkl.Dense(h, activation=sigma,kernel_initializer="orthogonal"),
      tfkl.Dense(3*p,kernel_initializer="orthogonal") # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
    ])

    tiledmask = tf.tile(xmask,[K,1])
    tiledmask_float = tf.cast(tiledmask,tf.float32)
    mask_not_float = tf.abs(-tf.cast(xmask,tf.float32))

    iota = tf.Variable(np.zeros([1,p]),dtype=tf.float32)
    tilediota = tf.tile(iota,[batch_size,1])
    iotax = x + tf.multiply(tilediota,mask_not_float)

    encoder = tfk.Sequential([
      tfkl.InputLayer(input_shape=[p,]),
      tfkl.Dense(h, activation=sigma,kernel_initializer="orthogonal"),
      tfkl.Dense(h, activation=sigma,kernel_initializer="orthogonal"),
      tfkl.Dense(3*d,kernel_initializer="orthogonal")
    ])

    out_encoder = encoder(iotax)
    q_zgivenxobs = tfd.Independent(distribution=tfd.StudentT(loc=out_encoder[..., :d], scale=tf.nn.softplus(out_encoder[..., d:(2*d)]), df=3 + tf.nn.softplus(out_encoder[..., (2*d):(3*d)])))
    zgivenx = q_zgivenxobs.sample(K)
    zgivenx_flat = tf.reshape(zgivenx,[K*batch_size,d])
    data_flat = tf.reshape(tf.tile(x,[K,1]),[-1,1])

    out_decoder = decoder(zgivenx_flat)
    all_means_obs_model = out_decoder[..., :p]
    all_scales_obs_model = tf.nn.softplus(out_decoder[..., p:(2*p)]) + 0.001
    all_degfreedom_obs_model = tf.nn.softplus(out_decoder[..., (2*p):(3*p)]) + 3
    all_log_pxgivenz_flat = tfd.StudentT(loc=tf.reshape(all_means_obs_model,[-1,1]),scale=tf.reshape(all_scales_obs_model,[-1,1]),df=tf.reshape(all_degfreedom_obs_model,[-1,1])).log_prob(data_flat)
    all_log_pxgivenz = tf.reshape(all_log_pxgivenz_flat,[K*batch_size,p])

    logpxobsgivenz = tf.reshape(tf.reduce_sum(tf.multiply(all_log_pxgivenz,tiledmask_float),1),[K,batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)

    miwae_loss = -tf.reduce_mean(tf.reduce_logsumexp(logpxobsgivenz + logpz - logq,0)) +tf.log(tf.cast(K,tf.float32))
    train_miss = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(miwae_loss)

    xgivenz = tfd.Independent(
          distribution=tfd.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model))

    imp_weights = tf.nn.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
    xms = tf.reshape(xgivenz.mean(),[K,batch_size,p])
    xm=tf.einsum('ki,kij->ij', imp_weights, xms)

    miwae_loss_train=np.array([])

    mse_train=np.array([])
    bs = 64 # batch size
    n_epochs = epochs
    xhat = np.copy(xhat_0) # This will be out imputed data matrix

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ep in range(1,n_epochs):
          perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
          batches_data = np.array_split(xhat_0[perm,], n/bs)
          batches_mask = np.array_split(mask[perm,], n/bs)
          for it in range(len(batches_data)):
              train_miss.run(feed_dict={x: batches_data[it], learning_rate: lr, K:20, xmask: batches_mask[it]}) # Gradient step
          if ep % 50 == 1 or ep == (n_epochs -1):
              losstrain = np.array([miwae_loss.eval(feed_dict={x: xhat_0, K:20, xmask: mask})]) # MIWAE bound evaluation
              miwae_loss_train = np.append(miwae_loss_train,-losstrain,axis=0)
              print('Epoch %g' %ep)
              print('MIWAE likelihood bound  %g' %-losstrain)
              for i in range(n): # We impute the observations one at a time for memory reasons
                  xhat[i,:][~mask[i,:]]=xm.eval(feed_dict={x: xhat_0[i,:].reshape([1,p]), K:1000, xmask: mask[i,:].reshape([1,p])})[~mask[i,:].reshape([1,p])]
              err = np.array(mse(xhat,data_x,mask))
              print('Imputation MSE  %g' %err)
              print('-----')

    return xhat, err

if __name__ == "__main__":
    main()