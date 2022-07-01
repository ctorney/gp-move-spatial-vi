"""
Copyright 2022 Colin Torney

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm

from tensorflow_probability.python.distributions import kullback_leibler

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

dtype = np.float64
NUM_LATENT = 2

class nsgpVI(tf.Module):
                                        
    def __init__(self,kernel_len,kernel_amp,n_inducing_points,inducing_index_points,dataset,num_training_points,
                 num_covars,prior_beta_len_means,prior_beta_amp_means,prior_beta_len_std,prior_beta_amp_std, segment_length,
                init_observation_noise_variance=0.0,num_sequential_samples=10,num_parallel_samples=10,jitter=1e-6):
               
        self.jitter=jitter
        self.num_covars = num_covars
        
        self.mean_len = tf.Variable([0.0], dtype=tf.float64, name='len_mean', trainable=1)
        self.mean_amp = tf.Variable([0.0], dtype=tf.float64, name='var_mean', trainable=1)
        
        self.beta_len_mean = tf.Variable(np.zeros(self.num_covars), dtype=tf.float64, name='beta_len_mean',trainable=0)
        self.beta_amp_mean = tf.Variable(np.zeros(self.num_covars), dtype=tf.float64, name='beta_amp_mean',trainable=0)
        
        self.beta_len_std = tfp.util.TransformedVariable(1e-1*np.ones(self.num_covars),tfb.Softplus(), dtype=tf.float64, name='beta_len_std',trainable=0)
        self.beta_amp_std = tfp.util.TransformedVariable(1e-1*np.ones(self.num_covars),tfb.Softplus(), dtype=tf.float64, name='beta_amp_std',trainable=0)
        
        self.inducing_index_points = tf.Variable(inducing_index_points,dtype=dtype,name='ind_points',trainable=0) #z's for lower level functions

        self.kernel_len = kernel_len
        self.kernel_amp = kernel_amp
        self.q_mu = tf.Variable(np.zeros((NUM_LATENT*n_inducing_points),dtype=dtype),name='ind_loc_post', trainable=0)
        self.q_scale = tfp.util.TransformedVariable([np.eye(NUM_LATENT*n_inducing_points, dtype=dtype)],
                                                    tfp.bijectors.FillScaleTriL(diag_shift=np.float64(1e-05)),
                                                    dtype=tf.float64, name='len_scale_post', trainable=0)
        
        self.q_sqrt = tf.linalg.LinearOperatorLowerTriangular(self.q_scale)
        
        self.variational_inducing_observations_posterior = tfd.MultivariateNormalLinearOperator(
                                                                      loc=self.q_mu,
                                                                      scale=self.q_sqrt) 

        
        self.inducing_prior = tfd.MultivariateNormalDiag(loc=tf.zeros((NUM_LATENT*n_inducing_points),dtype=tf.float64),name='ind_prior')
                        
        
        self.beta_len_prior = tfp.distributions.Normal(loc=prior_beta_len_means,scale=prior_beta_len_std)
        self.beta_amp_prior = tfp.distributions.Normal(loc=prior_beta_amp_means,scale=prior_beta_amp_std)
        
        self.M = n_inducing_points

        self.vi_param_list = [self.q_mu, #self.inducing_index_points, 
                              self.q_scale.non_trainable_variables[0], 
                              self.beta_len_mean,self.beta_amp_mean,
                              self.beta_len_std.non_trainable_variables[0],
                              self.beta_amp_std.non_trainable_variables[0]]
        
        self.vgp_observation_noise_variance = tf.Variable(init_observation_noise_variance,dtype=dtype,name='nv', trainable=1)
        
        self.num_sequential_samples=num_sequential_samples
        self.num_parallel_samples=num_parallel_samples
        
        self.dataset = dataset
        self.num_training_points=num_training_points
        
        n = segment_length-1
        # lower triangular matrix for calculating sums of matrix 
        self.LT = tfp.math.fill_triangular(tf.ones(n * (n+1) // 2, dtype=tf.float64))
        
        # noise matrix has a specific form based on the summation that is involved in the covariance matrix calc
        self.noise_matrix = tf.zeros((n,n), dtype=tf.float64)
        diag_part = 2.*np.ones(n)
        diag_part[0]=1.
        off_diag_part = -1.*np.ones(n-1)
        self.noise_matrix = tf.linalg.set_diag(self.noise_matrix,diag_part)
        self.noise_matrix = tf.linalg.set_diag(self.noise_matrix,off_diag_part,k=1)
        self.noise_matrix = tf.linalg.set_diag(self.noise_matrix,off_diag_part,k=-1)
    
    
    def optimize(self, BATCH_SIZE,  MAX_ITERS=100, lr=1e-2, threshold=None, window=1,patience=1):
        m_optimizer =  tf.keras.optimizers.Adam(learning_rate=lr)#,momentum=0.0)#False,epsilon=1e-03)#
        e_optimizer =  tf.keras.optimizers.Adam(learning_rate=lr)#,momentum=0.0)#False,epsilon=1e-03)#
        
                                                    
        @tf.function
        def m_train_step(inputs):
            t_train_batch, x_train_batch, predictor_batch = inputs
            kl_weight = tf.reduce_sum(tf.ones_like(t_train_batch))/self.num_training_points
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                loss = self.variational_loss(locations=x_train_batch,time_points=t_train_batch,predictor_values=predictor_batch, kl_weight=kl_weight) 
            grads = tape.gradient(loss, self.trainable_variables)
            m_optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return loss
        
        @tf.function
        def e_train_step(inputs):
            t_train_batch, x_train_batch, predictor_batch = inputs
            kl_weight = tf.reduce_sum(tf.ones_like(t_train_batch))/self.num_training_points
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.vi_param_list)
                loss = self.variational_loss(locations=x_train_batch,time_points=t_train_batch,predictor_values=predictor_batch, kl_weight=kl_weight) 
            grads = tape.gradient(loss, self.vi_param_list)
            e_optimizer.apply_gradients(zip(grads, self.vi_param_list))
            return loss
        
        pbar = tqdm(range(MAX_ITERS))
        loss_history = []

        current_window = None
        current_window_loss = None
        patience_count = patience
        for i in pbar:

            
            previous_window = current_window
            previous_window_loss = current_window_loss
            current_window = []
            current_window_loss = []

            for w in range(window):
                epoch_loss = 0
                batch_count=0    
                for batch in self.dataset:
                    batch_loss = e_train_step(batch).numpy()
                    batch_loss = m_train_step(batch).numpy()
                    epoch_loss += batch_loss
                    batch_count+=batch[0].shape[0]
                    pbar.set_description("Loss %f" % (epoch_loss/batch_count))
                    param_list = []
                    for param in self.vi_param_list:
                        param_list.append(param.numpy())
                    for param in self.trainable_variables:
                        param_list.append(param.numpy())
                    current_window.append(param_list)
                    current_window_loss.append([batch_loss/batch[0].shape[0]])
                loss_history.append(epoch_loss/batch_count)

            if threshold is None:
                continue
            if previous_window is not None:
                decrease = np.mean(np.array(previous_window_loss)) - np.mean(np.array(current_window_loss))
                if decrease <= threshold:
                    patience_count-=1
                else:
                    patience_count=patience
                if patience_count==0:
                    break
                
            
        #for i in range(len(self.vi_param_list)):
        #    self.vi_param_list[i].assign(np.mean(np.array([cw[i] for cw in current_window]),axis=0))
        #for i in range(len(self.vi_param_list), len(self.vi_param_list)+len(self.trainable_variables)):
        #    self.trainable_variables[i-len(self.vi_param_list)].assign(np.mean(np.array([cw[i] for cw in current_window]),axis=0))
        return loss_history




    def variational_loss(self,locations,time_points,predictor_values,kl_weight=1.0):
        
        kl_penalty = self.penalty()
        recon = self.surrogate_posterior_expected_log_likelihood(locations,time_points,predictor_values)
        return -recon  + kl_weight*kl_penalty

    
    def penalty(self):
        
        penalty = kullback_leibler.kl_divergence(self.variational_inducing_observations_posterior,self.inducing_prior) 
        
        penalty += tf.reduce_sum(tfp.distributions.kl_divergence(tfp.distributions.Normal(loc=self.beta_len_mean,scale=self.beta_len_std),self.beta_len_prior))
        penalty += tf.reduce_sum(tfp.distributions.kl_divergence(tfp.distributions.Normal(loc=self.beta_amp_mean,scale=self.beta_amp_std),self.beta_amp_prior))

        return penalty

    def surrogate_posterior_expected_log_likelihood(self,locations,time_points,predictor_values):

        len_vals, amp_vals = self.get_samples(locations,predictor_values,S=self.num_parallel_samples)   
        L = self.non_stat_vel(time_points, len_vals, amp_vals, tf.nn.softplus(self.vgp_observation_noise_variance)) # BxNxN
        
        centered_locations = locations[...,1:,:]-locations[...,0,None,:] #centered observations

        logpdf_K_x = tf.reduce_sum(tf.reduce_mean(tfd.MultivariateNormalTriL(scale_tril = L).log_prob((centered_locations[...,0])),axis=0))
        logpdf_K_y = tf.reduce_sum(tf.reduce_mean(tfd.MultivariateNormalTriL(scale_tril = L).log_prob((centered_locations[...,1])),axis=0))
        
        return logpdf_K_x + logpdf_K_y    
    
    def get_samples(self,locations,predictor_values,S=1):
        midpoints = 0.5*(locations[:,:-1]+locations[:,1:])
        mean, var = self.get_conditional(midpoints)
        samples = self.sample_conditional(mean, var, S)
    
        len_samples,amp_samples = tf.split(samples,NUM_LATENT,axis=2)
        
        # covariate samples
        B = tf.shape(mean)[0]
        
        z = tf.random.normal((S,B,self.num_covars),dtype=tf.float64)
        zlen = tf.expand_dims(tf.reshape(self.beta_len_mean,(1,1,-1)) + (tf.reshape(self.beta_len_std,(1,1,-1))*z),-1)
        scaled_predictors_len = tf.linalg.matmul(tf.expand_dims(predictor_values,0),zlen)
        
        z = tf.random.normal((S,B,self.num_covars),dtype=tf.float64)
        zamp = tf.expand_dims(tf.reshape(self.beta_amp_mean,(1,1,-1)) + (tf.reshape(self.beta_amp_std,(1,1,-1))*z),-1)
        scaled_predictors_amp = tf.linalg.matmul(tf.expand_dims(predictor_values,0),zamp)

        return tf.math.exp(self.mean_len + scaled_predictors_len + len_samples), \
                tf.math.exp(self.mean_amp + scaled_predictors_amp + amp_samples)
    
    def get_conditional(self, X):
        
        Z = self.inducing_index_points 
        M = self.M

        Lm_len = tf.linalg.LinearOperatorFullMatrix(self.kernel_len.matrix(Z,Z) + self.jitter * tf.eye(M, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True).cholesky()
        Lm_amp = tf.linalg.LinearOperatorFullMatrix(self.kernel_amp.matrix(Z,Z) + self.jitter * tf.eye(M, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True).cholesky()

        Kmn_len = tf.linalg.LinearOperatorFullMatrix(self.kernel_len.matrix(Z, X),is_positive_definite=True,is_self_adjoint=True)
        Kmn_amp = tf.linalg.LinearOperatorFullMatrix(self.kernel_amp.matrix(Z, X),is_positive_definite=True,is_self_adjoint=True)

        Lm_len_inv_Kmn = Lm_len.solve(Kmn_len)
        Lm_amp_inv_Kmn = Lm_amp.solve(Kmn_amp)
        Lm_inv_Kmn = tf.linalg.LinearOperatorBlockDiag([Lm_len_inv_Kmn,Lm_amp_inv_Kmn])

        mean_f = tf.expand_dims(Lm_inv_Kmn.matvec(self.q_mu, adjoint=True),-1)

        Lm_inv_Kmn_q = Lm_inv_Kmn.matmul(self.q_sqrt, adjoint=True)
        Lm_inv_Kmn_q2 = Lm_inv_Kmn_q.matmul(Lm_inv_Kmn_q,adjoint_arg=True)

        Knn_len = tf.linalg.LinearOperatorFullMatrix(self.kernel_len.matrix(X, X),is_positive_definite=True,is_self_adjoint=True)
        Knn_amp = tf.linalg.LinearOperatorFullMatrix(self.kernel_amp.matrix(X, X),is_positive_definite=True,is_self_adjoint=True)

        Knn = tf.linalg.LinearOperatorBlockDiag([Knn_len,Knn_amp])

        Lm_len_inv_Kmn2 = Lm_len_inv_Kmn.matmul(Lm_len_inv_Kmn,adjoint=True)
        Lm_amp_inv_Kmn2 = Lm_amp_inv_Kmn.matmul(Lm_amp_inv_Kmn,adjoint=True)
        Lm_inv_Kmn2 = tf.linalg.LinearOperatorBlockDiag([Lm_len_inv_Kmn2,Lm_amp_inv_Kmn2])

        covar_f = Lm_inv_Kmn_q2.to_dense() + Knn.to_dense() - Lm_inv_Kmn2.to_dense()

        return mean_f, covar_f

    def get_marginal(self, X):

        tf.debugging.assert_rank(X,3,message="get_marginal expects a batch of locations. Add first dimension of size 1 if processing a single batch" )

        mean_f, covar_f = self.get_conditional(X)

        covar_f = tf.linalg.diag_part(covar_f)
        mean_list = tf.split(mean_f,NUM_LATENT,axis=1)
        var_list = tf.split(covar_f,NUM_LATENT,axis=1)

        return mean_list, var_list


       

    def sample_conditional(self, mean, var, S=1):
        # mean BxNx1
        # var BxNxN
        # returns SxBxNx1
        B = tf.shape(mean)[0]
        N = tf.shape(mean)[1]
        z = tf.random.normal((S,B,N,1),dtype=tf.float64)
        
        I = self.jitter**1 * tf.eye(N, dtype=tf.float64) #NN
        chol = tf.linalg.cholesky(var + I)  # BNN

        samples = tf.expand_dims(mean,0) + tf.matmul(chol, z)#[:, :, :, 0]  # BSN1
        return samples

    def non_stat_vel(self,T,lengthscales, var, obs_noise):
        
        """Non-stationary integrated Matern12 kernel"""
        stddev = tf.math.sqrt(var)
        
        sigma_ = stddev[...,0,None]
        len_ = lengthscales[...,0,None]
        Ls = tf.square(len_)

        L = tf.math.sqrt(0.5*(Ls + tf.linalg.matrix_transpose(Ls)))

        prefactL = tf.math.sqrt(tf.matmul(len_, len_, transpose_b=True))
        prefactV = tf.matmul(sigma_, sigma_,transpose_b=True)

        zeta = tf.math.multiply(prefactV,tf.math.divide(prefactL,L))


        tpq1 = tf.math.exp(tf.math.divide(-tf.math.abs(tf.linalg.matrix_transpose(T[:,:-1]) - T[:,1:]),L))
        tp1q1 = tf.math.exp(tf.math.divide(-tf.math.abs(tf.linalg.matrix_transpose(T[:,1:]) - T[:,1:]),L))
        tpq = tf.math.exp(tf.math.divide(-tf.math.abs(tf.linalg.matrix_transpose(T[:,:-1]) - T[:,:-1]),L))
        tp1q = tf.math.exp(tf.math.divide(-tf.math.abs(tf.linalg.matrix_transpose(T[:,1:]) - T[:,:-1]),L))


        Epq_grid = tpq1-tp1q1-tpq+tp1q
        Epq_grid = (L**2)*Epq_grid

        Epq_grid = tf.linalg.set_diag(Epq_grid,(tf.linalg.diag_part(Epq_grid)) + 2.0*len_[...,0]*((T[:,1:,0])-(T[:,:-1,0])))
        Epq_grid = zeta*Epq_grid


        

        M = Epq_grid + self.noise_matrix*obs_noise

        cholesky_factor = tf.linalg.matmul(self.LT,tf.linalg.cholesky(M))

        
        return cholesky_factor
    
