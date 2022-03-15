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

from utils.gradient_accumulator import GradientAccumulator

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

dtype = np.float64
NUM_LATENT = 2

class nsgpVI(tf.Module):
                                        
    def __init__(self,
                 kernel_len,
                 kernel_amp,
                 n_inducing_points,
                 inducing_index_points,
                 dataset,
                 num_training_points, 
                 kernel_mf=None,
                 init_observation_noise_variance=1e-2,
                 num_sequential_samples=10,
                 num_parallel_samples=10,
                 jitter=1e-6):
               
        self.jitter=jitter
        
        #self.L = domain_size
        self.mean_len = tf.Variable([0], dtype=tf.float64, name='len_mean', trainable=1)
        self.mean_amp = tf.Variable([0], dtype=tf.float64, name='var_mean', trainable=1)
        self.inducing_index_points = tf.Variable(inducing_index_points,dtype=dtype,name='ind_points',trainable=1) #z's for lower level functions

        self.kernel_len = kernel_len
        self.kernel_amp = kernel_amp
        self.kernel_mf = kernel_mf
        
        #parameters for variational distribution for len,phi(l_z) and var,phi(sigma_z)
        self.q_mu = tf.Variable(np.zeros((NUM_LATENT*n_inducing_points),dtype=dtype),name='ind_loc_post')
        self.len_scale = tfp.util.TransformedVariable([np.eye(n_inducing_points, dtype=dtype)],
                                                      tfp.bijectors.FillScaleTriL(diag_shift=np.float64(1e-05)),dtype=tf.float64, name='len_scale_post')
        self.amp_scale = tfp.util.TransformedVariable([np.eye(n_inducing_points, dtype=dtype)],
                                                      tfp.bijectors.FillScaleTriL(diag_shift=np.float64(1e-05)),dtype=tf.float64, name='amp_scale_post')
        self.cc_scale = tf.Variable([np.zeros((n_inducing_points),dtype=dtype)],name='cc_scale',trainable=1)

        len_op = tf.linalg.LinearOperatorLowerTriangular(self.len_scale)
        amp_op = tf.linalg.LinearOperatorLowerTriangular(self.amp_scale)
        cc_op = tf.linalg.LinearOperatorDiag(self.cc_scale)
        
        self.q_sqrt = tf.linalg.LinearOperatorBlockLowerTriangular([[len_op],[cc_op,amp_op]])    
        #approximation to the posterior: phi(l_z)
        self.variational_inducing_observations_posterior = tfd.MultivariateNormalLinearOperator(
                                                                      loc=self.q_mu,
                                                                      scale=self.q_sqrt) 
        
        if self.kernel_mf is not None:
            self.q_mu_x = tf.Variable(np.zeros((n_inducing_points),dtype=dtype),name='q_mu_x')
            self.x_scale = tfp.util.TransformedVariable([np.eye(n_inducing_points, dtype=dtype)],
                                                          tfp.bijectors.FillScaleTriL(diag_shift=np.float64(1e-05)),dtype=tf.float64, name='x_scale')
            self.q_sqrt_x = tf.linalg.LinearOperatorLowerTriangular(self.x_scale)

            self.q_mu_y = tf.Variable(np.zeros((n_inducing_points),dtype=dtype),name='q_mu_y')
            self.y_scale = tfp.util.TransformedVariable([np.eye(n_inducing_points, dtype=dtype)],
                                                          tfp.bijectors.FillScaleTriL(diag_shift=np.float64(1e-05)),dtype=tf.float64, name='y_scale')
            self.q_sqrt_y = tf.linalg.LinearOperatorLowerTriangular(self.y_scale)


            self.variational_inducing_observations_posterior_x = tfd.MultivariateNormalLinearOperator(
                                                                          loc=self.q_mu_x,
                                                                          scale=self.q_sqrt_x) 
            self.variational_inducing_observations_posterior_y = tfd.MultivariateNormalLinearOperator(
                                                                          loc=self.q_mu_y,
                                                                          scale=self.q_sqrt_y) 

            self.inducing_prior_xy = tfd.MultivariateNormalDiag(loc=tf.zeros((n_inducing_points),dtype=tf.float64),name='ind_prior')

        self.inducing_prior = tfd.MultivariateNormalDiag(loc=tf.zeros((NUM_LATENT*n_inducing_points),dtype=tf.float64),name='ind_prior')
        self.M = n_inducing_points
        
        
        
        self.obs_max = tf.Variable([0.01], dtype=tf.float64, name='obs_max', trainable=False)
        
        self.vgp_observation_noise_variance = tf.Variable(0.0,dtype=dtype,name='nv', trainable=1)

        self.num_sequential_samples=num_sequential_samples
        self.num_parallel_samples=num_parallel_samples
        
        self.dataset = dataset
        self.num_training_points=num_training_points
        

    def optimize(self, BATCH_SIZE, SEG_LENGTH, NUM_EPOCHS=100):


        strategy = tf.distribute.MirroredStrategy()
        dist_dataset = strategy.experimental_distribute_dataset(self.dataset)

        initial_learning_rate = 1e-2
        steps_per_epoch = self.num_training_points//(BATCH_SIZE*SEG_LENGTH)
        learning_rate = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,decay_steps=steps_per_epoch,decay_rate=0.99,staircase=True)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=initial_learning_rate,momentum=0.9)#,centered=False,epsilon=1e-03)
        #optimizer = tf.keras.optimizers.Adadelta(learning_rate=initial_learning_rate)

        #optimizer = tf.keras.optimizers.Nadam(learning_rate=initial_learning_rate)#, beta_1=0.0, beta_2=0.6, epsilon=1e-1, amsgrad=False)

        accumulator = GradientAccumulator()

        def train_step(inputs):
            t_train_batch, x_train_batch, predictor_batch = inputs
            kl_weight = tf.reduce_sum(tf.ones_like(t_train_batch))/self.num_training_points

            with tf.GradientTape(watch_accessed_variables=True) as tape:
                loss = self.variational_loss(locations=x_train_batch,time_points=t_train_batch,predictor_values=predictor_batch, kl_weight=kl_weight) 
            grads = tape.gradient(loss, self.trainable_variables)
            return loss, grads

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses, per_replica_grads = strategy.run(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_grads, axis=None)

        pbar = tqdm(range(NUM_EPOCHS))
        loss_history = np.zeros((NUM_EPOCHS))
        len_history = np.zeros((NUM_EPOCHS))


        for i in pbar:
            batch_count=0    
            epoch_loss = 0.0
            for batch in self.dataset:
                batch_loss = 0.0
                for s in range(self.num_sequential_samples):
                    loss, grads = distributed_train_step(batch)
                    # accumulate the loss and gradient
                    accumulator(grads)
                    batch_loss += loss.numpy()
               
                grads = accumulator.gradients
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                batch_loss/=self.num_sequential_samples
                accumulator.reset()
                    
                epoch_loss+=batch_loss
                batch_count+=batch[0].shape[0]
                #pbar.set_description("Loss %f" % (epoch_loss/batch_count))
                pbar.set_description("Loss %f, klen_l %f, kamp_l %f, obs %f" % (epoch_loss/batch_count, self.kernel_len.length_scale.numpy(), self.kernel_amp.length_scale.numpy(),(self.obs_max*tf.nn.sigmoid(self.vgp_observation_noise_variance)).numpy()))
            loss_history[i] = epoch_loss/batch_count
            len_history[i] = self.kernel_len.length_scale.numpy()
            #print(epoch_loss)

        return loss_history, len_history



    def variational_loss(self,locations,time_points,predictor_values,kl_weight=1.0):
        
        kl_penalty = self.penalty()
        recon = self.surrogate_posterior_expected_log_likelihood(locations,time_points,predictor_values)
        return -recon  + kl_weight*kl_penalty

    
    def penalty(self):
        
        penalty = 2.0*kullback_leibler.kl_divergence(self.variational_inducing_observations_posterior,self.inducing_prior) 
        
        if self.kernel_mf is not None:
            penalty += kullback_leibler.kl_divergence(self.variational_inducing_observations_posterior_x,self.inducing_prior_xy) 
            penalty += kullback_leibler.kl_divergence(self.variational_inducing_observations_posterior_y,self.inducing_prior_xy) 
        
        return penalty

    def surrogate_posterior_expected_log_likelihood(self,locations,time_points,predictor_values):
        len_vals, amp_vals = self.get_samples(locations,predictor_values,S=self.num_parallel_samples)   

        K = self.non_stat_vel(time_points, len_vals, amp_vals) # BxNxN
        K = K + (tf.eye(tf.shape(K)[-1], dtype=tf.float64) * ((self.obs_max * tf.nn.sigmoid(self.vgp_observation_noise_variance))+self.jitter))

        x_mean_vels, y_mean_vels = vgp.get_mean_samples(locations,S=vgp.num_parallel_samples)   

        centered_locations = locations[...,:,:]-locations[...,0,None,:] #centered observations

        timedifference = tf.expand_dims((time_points[:,1:])-(time_points[:,:-1]),0)

        delta_x = x_mean_vels*timedifference
        delta_y = y_mean_vels*timedifference

        mean_x = tf.expand_dims(centered_locations,0)[...,:-1,0,None]+delta_x
        mean_y = tf.expand_dims(centered_locations,0)[...,:-1,1,None]+delta_y

        logpdf_K_x = tf.reduce_sum(tf.reduce_mean(tfd.MultivariateNormalTriL(loc=mean_x[...,0],scale_tril = tf.linalg.cholesky(K)).log_prob((centered_locations[...,1:,0])),axis=0))
        logpdf_K_y = tf.reduce_sum(tf.reduce_mean(tfd.MultivariateNormalTriL(loc=mean_y[...,0],scale_tril = tf.linalg.cholesky(K)).log_prob((centered_locations[...,1:,1])),axis=0))

        #centered_locations = locations[...,1:,:]-locations[...,0,None,:] #centered observations

        #logpdf_K_x = tf.reduce_sum(tf.reduce_mean(tfd.MultivariateNormalTriL(scale_tril = tf.linalg.cholesky(K)).log_prob((centered_locations[...,0])),axis=0))
        #logpdf_K_y = tf.reduce_sum(tf.reduce_mean(tfd.MultivariateNormalTriL(scale_tril = tf.linalg.cholesky(K)).log_prob((centered_locations[...,1])),axis=0))
        
        return logpdf_K_x + logpdf_K_y    
    
    def get_samples(self,locations,predictor_values,S=1):
        midpoints = 0.5*(locations[:,:-1]+locations[:,1:])
        mean, var = self.get_conditional(midpoints)
        samples = self.sample_conditional(mean, var, S)
    
        len_samples,amp_samples = tf.split(samples,NUM_LATENT,axis=2)
        
        return tf.math.exp(self.mean_len + len_samples), tf.math.exp(self.mean_amp + amp_samples)
    
    
    def get_mean_samples(self,locations,S=1):
        midpoints = 0.5*(locations[:,:-1]+locations[:,1:])
        meanx, meany, varx, vary = self.get_conditional_mean(midpoints)
        samples_x = self.sample_conditional(meanx, varx, S)
        samples_y = self.sample_conditional(meany, vary, S)
    
        return samples_x, samples_y
    
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
    
    
    def get_conditional_mean(self, X):
        
        Z = self.inducing_index_points 
        M = self.M

        Lm = tf.linalg.LinearOperatorFullMatrix(self.kernel_mf.matrix(Z,Z) + self.jitter * tf.eye(M, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True).cholesky()
        
        Kmn = tf.linalg.LinearOperatorFullMatrix(self.kernel_mf.matrix(Z, X),is_positive_definite=True,is_self_adjoint=True)

        Lm_inv_Kmn = Lm.solve(Kmn)
        
        mean_fx = tf.expand_dims(Lm_inv_Kmn.matvec(self.q_mu_x, adjoint=True),-1)
        mean_fy = tf.expand_dims(Lm_inv_Kmn.matvec(self.q_mu_y, adjoint=True),-1)

        Lm_inv_Kmn_qx = Lm_inv_Kmn.matmul(self.q_sqrt_x, adjoint=True)
        Lm_inv_Kmn_q2x = Lm_inv_Kmn_qx.matmul(Lm_inv_Kmn_qx,adjoint_arg=True)

        Lm_inv_Kmn_qy = Lm_inv_Kmn.matmul(self.q_sqrt_y, adjoint=True)
        Lm_inv_Kmn_q2y = Lm_inv_Kmn_qy.matmul(Lm_inv_Kmn_qy,adjoint_arg=True)

        Knn = tf.linalg.LinearOperatorFullMatrix(self.kernel_mf.matrix(X, X),is_positive_definite=True,is_self_adjoint=True)
        Lm_inv_Kmn2 = Lm_inv_Kmn.matmul(Lm_inv_Kmn,adjoint=True)
        
        covar_fx = Lm_inv_Kmn_q2x.to_dense() + Knn.to_dense() - Lm_inv_Kmn2.to_dense()
        covar_fy = Lm_inv_Kmn_q2y.to_dense() + Knn.to_dense() - Lm_inv_Kmn2.to_dense()

        return mean_fx, mean_fy, covar_fx, covar_fy

    def get_marginal(self, X):

        tf.debugging.assert_rank(X,3,message="get_marginal expects a batch of locations. Add first dimension of size 1 if processing a single batch" )

        mean_f, covar_f = self.get_conditional(X)

        covar_f = tf.linalg.diag_part(covar_f)
        mean_list = tf.split(mean_f,NUM_LATENT,axis=1)
        var_list = tf.split(covar_f,NUM_LATENT,axis=1)

        return mean_list, var_list

    def get_marginal_mean(self, X):

        tf.debugging.assert_rank(X,3,message="get_marginal expects a batch of locations. Add first dimension of size 1 if processing a single batch" )

        mean_fx, mean_fy, covar_fx, covar_fy = self.get_conditional_mean(X)

        covar_fx = tf.linalg.diag_part(covar_fx)
        covar_fy = tf.linalg.diag_part(covar_fy)
        
        return mean_fx, mean_fy, covar_fx, covar_fy
       

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

    def non_stat_vel(self,T,lengthscales, var):
        
        """Non-stationary integrated Matern12 kernel"""
        stddev = tf.math.sqrt(var)
        sigma_ = stddev[...,0,None]# + stddev[...,1:,0,None])
        len_ = lengthscales[...,0,None]# + lengthscales[...,1:,0,None])

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


        K = tf.math.cumsum(tf.math.cumsum(Epq_grid,axis=-2,exclusive=False),axis=-1,exclusive=False)
        
        return K
    