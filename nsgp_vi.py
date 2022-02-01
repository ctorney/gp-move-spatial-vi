"""
Copyright 2021 Colin Torney

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
                                        
    def __init__(self,kernel_len,kernel_amp,n_inducing_points,inducing_index_points,dataset,num_training_points, init_observation_noise_variance=1e-2,num_sequential_samples=10,num_parallel_samples=10,jitter=1e-6):
               
        self.jitter=jitter
        
        #self.L = domain_size
        self.mean_len = tf.Variable([0.0], dtype=tf.float64, name='len_mean', trainable=True)
        self.mean_amp = tf.Variable([0.0], dtype=tf.float64, name='var_mean', trainable=True)
        
        self.amp_inducing_index_points = tf.Variable(inducing_index_points,dtype=dtype,name='amp_ind_points',trainable=0) #z's for amplitude
        self.len_inducing_index_points = tf.Variable(inducing_index_points,dtype=dtype,name='len_ind_points',trainable=0) #z's for len

        self.kernel_len = kernel_len
        self.kernel_amp = kernel_amp
        
        #parameters for variational distribution for len,phi(l_z) and var,phi(sigma_z)
        self.variational_inducing_observations_loc = tf.Variable(np.zeros((NUM_LATENT*n_inducing_points),dtype=dtype),name='ind_loc_post')
        #self.variational_inducing_observations_scale = tfp.util.TransformedVariable(np.eye(NUM_LATENT*n_inducing_points, dtype=dtype),tfp.bijectors.FillScaleTriL(diag_shift=np.float64(1e-05)),dtype=tf.float64, name='ind_scale_post', trainable=1)
        self.variational_inducing_observations_scale = tfp.util.TransformedVariable(np.float64(1e-01)*2.*np.eye(NUM_LATENT*n_inducing_points, dtype=dtype),tfp.bijectors.FillScaleTriL(diag_shift=np.float64(1e-05)),dtype=tf.float64, name='ind_scale_post', trainable=1)

        
        #approximation to the posterior: phi(l_z)
        self.variational_inducing_observations_posterior = tfd.MultivariateNormalLinearOperator(
                                                                      loc=self.variational_inducing_observations_loc,
                                                                      scale=tf.linalg.LinearOperatorLowerTriangular(self.variational_inducing_observations_scale))

        #p(l_z)
        self.inducing_prior = tfd.MultivariateNormalDiag(loc=tf.zeros((NUM_LATENT*n_inducing_points),dtype=tf.float64),name='ind_prior')
        
        self.vgp_observation_noise_variance = tf.Variable(np.log(np.exp(init_observation_noise_variance)-1),dtype=dtype,name='nv', trainable=1)

        self.num_sequential_samples=num_sequential_samples
        self.num_parallel_samples=num_parallel_samples
        
        self.dataset = dataset
        self.num_training_points=num_training_points
        

    def optimize(self, BATCH_SIZE, SEG_LENGTH, NUM_EPOCHS=100):


        strategy = tf.distribute.MirroredStrategy()
        dist_dataset = strategy.experimental_distribute_dataset(self.dataset)

        initial_learning_rate = 1e-2
        steps_per_epoch = 50#self.num_training_points//(BATCH_SIZE*SEG_LENGTH)
        learning_rate = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,decay_steps=steps_per_epoch,decay_rate=0.99,staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_2=0.99)
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
            #return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), [strategy.reduce(tf.distribute.ReduceOp.SUM, prg, axis=None) for prg in per_replica_grads]

        pbar = tqdm(range(NUM_EPOCHS))
        loss_history = np.zeros((NUM_EPOCHS))

        for i in pbar:
            batch_count=0    
            epoch_loss = 0.0
            for batch in self.dataset:
                batch_loss = 0.0
                for s in range(self.num_sequential_samples):
                    #try:
                    loss, grads = distributed_train_step(batch)
                    # accumulate the loss and gradient
                    accumulator(grads)
                    batch_loss += loss.numpy()
                    #except:
                    #    pass
                #if accumulator.step:
                grads = accumulator.gradients
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                batch_loss/=self.num_sequential_samples
                accumulator.reset()
                    
                epoch_loss+=batch_loss
                batch_count+=1
                pbar.set_description("Loss %f, klen_l %f, kamp_l %f" % (epoch_loss/batch_count, self.kernel_len.length_scale.numpy(),self.kernel_len.amplitude.numpy()))
            loss_history[i] = epoch_loss#/batch_count
            #print(epoch_loss)

        return loss_history



    def variational_loss(self,locations,time_points,predictor_values,kl_weight=1.0):
        
        kl_penalty = self.surrogate_posterior_kl_divergence_prior()
        recon = self.surrogate_posterior_expected_log_likelihood(locations,time_points,predictor_values)
        #return (-recon)
        return (-recon + kl_weight*kl_penalty)

    
    def surrogate_posterior_kl_divergence_prior(self):
        #return -self.inducing_prior.log_prob(self.variational_inducing_observations_loc)
        return kullback_leibler.kl_divergence(self.variational_inducing_observations_posterior,self.inducing_prior) 

    def surrogate_posterior_kl_divergence_prior2(self):

        Z_amp = self.amp_inducing_index_points 
        Z_len = self.len_inducing_index_points 
        kernel_amp = self.kernel_amp
        kernel_len = self.kernel_len

        f = self.variational_inducing_observations_loc
        ipmvn  = tfd.MultivariateNormalDiag(loc=tf.zeros_like(f),name='ind_prior')
        M = tf.shape(f)[0]

        Kmm_amp = tf.linalg.LinearOperatorFullMatrix(kernel_amp.matrix(Z_amp,Z_amp) + self.jitter * tf.eye(M//2, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True)
        Kmm_len = tf.linalg.LinearOperatorFullMatrix(kernel_len.matrix(Z_len,Z_len) + self.jitter * tf.eye(M//2, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True)


        return -ipmvn.log_prob(f)+0.5*tf.linalg.logdet(Kmm_amp)+0.5*tf.linalg.logdet(Kmm_len)
#         Z_amp = self.amp_inducing_index_points 
#         Z_len = self.len_inducing_index_points 

#         kernel_amp = self.kernel_amp
#         kernel_len = self.kernel_len

#         f = self.variational_inducing_observations_loc
#         q_sqrt = self.variational_inducing_observations_scale

#         M = tf.shape(f)[0]

#         Kmm_amp = tf.linalg.LinearOperatorFullMatrix(kernel_amp.matrix(Z_amp,Z_amp) + self.jitter * tf.eye(M//2, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True)
#         Kmm_len = tf.linalg.LinearOperatorFullMatrix(kernel_len.matrix(Z_len,Z_len) + self.jitter * tf.eye(M//2, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True)
#         Kmm = tf.linalg.LinearOperatorBlockDiag([Kmm_len,Kmm_amp])


#         Kmn_amp = tf.linalg.LinearOperatorFullMatrix(kernel_amp.matrix(Z_amp, Z_amp),is_positive_definite=True,is_self_adjoint=True)
#         Kmn_len = tf.linalg.LinearOperatorFullMatrix(kernel_len.matrix(Z_len, Z_len),is_positive_definite=True,is_self_adjoint=True)
#         Kmn = tf.linalg.LinearOperatorBlockDiag([Kmn_len,Kmn_amp])


#         Knn_amp = tf.linalg.LinearOperatorFullMatrix(kernel_amp.matrix(Z_amp, Z_amp),is_positive_definite=True,is_self_adjoint=True)
#         Knn_len = tf.linalg.LinearOperatorFullMatrix(kernel_len.matrix(Z_len, Z_len),is_positive_definite=True,is_self_adjoint=True)
#         Knn = tf.linalg.LinearOperatorBlockDiag([Knn_len,Knn_amp])

#         mean,var = self.full_conditional(Kmn,Kmm,Knn,f,q_sqrt)

#         #Z_len
#         mvn = tfd.MultivariateNormalLinearOperator( loc=tf.zeros_like(f), scale=Kmm.cholesky())

#         mvn2 = tfd.MultivariateNormalTriL(loc=mean[0,:,0], scale_tril=tf.linalg.cholesky(var[0]))

#         return tfp.distributions.kl_divergence(mvn2,mvn)
    def surrogate_posterior_expected_log_likelihood(self,locations,time_points,predictor_values):

        len_vals, amp_vals = self.get_samples(predictor_values,S=self.num_parallel_samples)   
        K = self.non_stat_vel(time_points, len_vals, amp_vals) # BxNxN
        K = K + (tf.eye(tf.shape(K)[-1], dtype=tf.float64) * ((tf.nn.softplus(self.vgp_observation_noise_variance))+self.jitter))

        
        centered_locations = locations[...,1:,:]-locations[...,0,None,:] #centered observations

        logpdf_K_x = tf.reduce_sum(tf.reduce_mean(tfd.MultivariateNormalTriL(scale_tril = tf.linalg.cholesky(K)).log_prob((centered_locations[...,0])),axis=0))
        logpdf_K_y = tf.reduce_sum(tf.reduce_mean(tfd.MultivariateNormalTriL(scale_tril = tf.linalg.cholesky(K)).log_prob((centered_locations[...,1])),axis=0))
        
        return logpdf_K_x + logpdf_K_y    
    
    def get_samples(self,predictor_values,S=1):
        mean, var = self.get_conditional(predictor_values)
        samples = self.sample_conditional(mean, var, S)
    
        amp_samples,len_samples = tf.split(samples,NUM_LATENT,axis=2)
        
        return tf.math.exp(self.mean_len + len_samples), tf.math.exp(self.mean_amp + amp_samples)
    
    def get_conditional(self, observation_index_points):
        
        Xnew = observation_index_points

        Z_amp = self.amp_inducing_index_points 
        Z_len = self.len_inducing_index_points 

        #Z=tf.linalg.LinearOperatorFullMatrix(Z)


        kernel_amp = self.kernel_amp
        kernel_len = self.kernel_len

        f = self.variational_inducing_observations_loc
        q_sqrt = self.variational_inducing_observations_scale


        
        M = tf.shape(f)[0]
        Kmm_amp = tf.linalg.LinearOperatorFullMatrix(kernel_amp.matrix(Z_amp,Z_amp) + self.jitter * tf.eye(M//2, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True)
        Kmm_len = tf.linalg.LinearOperatorFullMatrix(kernel_len.matrix(Z_len,Z_len) + self.jitter * tf.eye(M//2, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True)

        #Kmm = tf.linalg.LinearOperatorBlockDiag([Kmm_len,Kmm_amp])
        Kmm = tf.linalg.LinearOperatorBlockDiag([Kmm_amp,Kmm_len])


        Kmn_amp = tf.linalg.LinearOperatorFullMatrix(kernel_amp.matrix(Z_amp, Xnew),is_positive_definite=True,is_self_adjoint=True)
        Kmn_len = tf.linalg.LinearOperatorFullMatrix(kernel_len.matrix(Z_len, Xnew),is_positive_definite=True,is_self_adjoint=True)

        #Kmn = tf.linalg.LinearOperatorBlockDiag([Kmn_len,Kmn_amp])
        Kmn = tf.linalg.LinearOperatorBlockDiag([Kmn_amp,Kmn_len])

        Knn = kernel_amp.matrix(Xnew,Xnew)
        Knn_amp = tf.linalg.LinearOperatorFullMatrix(kernel_amp.matrix(Xnew, Xnew),is_positive_definite=True,is_self_adjoint=True)
        Knn_len = tf.linalg.LinearOperatorFullMatrix(kernel_len.matrix(Xnew, Xnew),is_positive_definite=True,is_self_adjoint=True)
        #Knn = tf.linalg.LinearOperatorBlockDiag([Knn_len,Knn_amp])
        Knn = tf.linalg.LinearOperatorBlockDiag([Knn_amp,Knn_len])

        mean,var = self.full_conditional(Kmn.to_dense(),Kmm.to_dense(),Knn.to_dense(),f,q_sqrt)
        
        return mean, var
    def get_conditional_lo(self, predictor_values):
        
        Xnew = predictor_values

        Z_amp = self.amp_inducing_index_points 
        Z_len = self.len_inducing_index_points 

        kernel_amp = self.kernel_amp
        kernel_len = self.kernel_len

        f = self.variational_inducing_observations_loc
        q_sqrt = self.variational_inducing_observations_scale

        M = tf.shape(f)[0]

        Kmm_amp = tf.linalg.LinearOperatorFullMatrix(kernel_amp.matrix(Z_amp,Z_amp) + self.jitter * tf.eye(M//2, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True)
        Kmm_len = tf.linalg.LinearOperatorFullMatrix(kernel_len.matrix(Z_len,Z_len) + self.jitter * tf.eye(M//2, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True)
        Kmm = tf.linalg.LinearOperatorBlockDiag([Kmm_len,Kmm_amp])


        Kmn_amp = tf.linalg.LinearOperatorFullMatrix(kernel_amp.matrix(Z_amp, Xnew),is_positive_definite=True,is_self_adjoint=True)
        Kmn_len = tf.linalg.LinearOperatorFullMatrix(kernel_len.matrix(Z_len, Xnew),is_positive_definite=True,is_self_adjoint=True)
        Kmn = tf.linalg.LinearOperatorBlockDiag([Kmn_len,Kmn_amp])


        Knn_amp = tf.linalg.LinearOperatorFullMatrix(kernel_amp.matrix(Xnew, Xnew),is_positive_definite=True,is_self_adjoint=True)
        Knn_len = tf.linalg.LinearOperatorFullMatrix(kernel_len.matrix(Xnew, Xnew),is_positive_definite=True,is_self_adjoint=True)
        Knn = tf.linalg.LinearOperatorBlockDiag([Knn_len,Knn_amp])

        mean,var = self.full_conditional_lo(Kmn,Kmm,Knn,f,q_sqrt)
        
        return mean, var

    def get_marginal(self, predictor_values):
        
        Xnew = predictor_values


        Z_amp = self.amp_inducing_index_points 
        Z_len = self.len_inducing_index_points 

        kernel_amp = self.kernel_amp
        kernel_len = self.kernel_len

        f = self.variational_inducing_observations_loc
        q_sqrt = self.variational_inducing_observations_scale


        M = tf.shape(f)[0]

        Kmm_amp = tf.linalg.LinearOperatorFullMatrix(kernel_amp.matrix(Z_amp,Z_amp) + self.jitter * tf.eye(M//2, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True)
        Kmm_len = tf.linalg.LinearOperatorFullMatrix(kernel_len.matrix(Z_len,Z_len) + self.jitter * tf.eye(M//2, dtype=tf.float64),is_positive_definite=True,is_self_adjoint=True)
        #Kmm = tf.linalg.LinearOperatorBlockDiag([Kmm_len,Kmm_amp])
        Kmm = tf.linalg.LinearOperatorBlockDiag([Kmm_amp,Kmm_len])


        Kmn_amp = tf.linalg.LinearOperatorFullMatrix(kernel_amp.matrix(Z_amp, Xnew),is_positive_definite=True,is_self_adjoint=True)
        Kmn_len = tf.linalg.LinearOperatorFullMatrix(kernel_len.matrix(Z_len, Xnew),is_positive_definite=True,is_self_adjoint=True)
        #Kmn = tf.linalg.LinearOperatorBlockDiag([Kmn_len,Kmn_amp])
        Kmn = tf.linalg.LinearOperatorBlockDiag([Kmn_amp,Kmn_len])


        Knn_amp = tf.linalg.LinearOperatorFullMatrix(kernel_amp.matrix(Xnew, Xnew),is_positive_definite=True,is_self_adjoint=True)
        Knn_len = tf.linalg.LinearOperatorFullMatrix(kernel_len.matrix(Xnew, Xnew),is_positive_definite=True,is_self_adjoint=True)
        #Knn = tf.linalg.LinearOperatorBlockDiag([Knn_len,Knn_amp])
        Knn = tf.linalg.LinearOperatorBlockDiag([Knn_amp,Knn_len])

        mean, var = self.marginal(Kmn.to_dense(),Kmm.to_dense(),Knn.to_dense(),f,q_sqrt)
        
        mean_list = tf.split(mean,NUM_LATENT,axis=0)
        var_list = tf.split(var,NUM_LATENT,axis=0)

        return mean_list, var_list
        

    def sample_conditional(self, mean, var, S=1):
        # mean BxNx1
        # var BxNxN
        # returns SxBxNx1
        B = tf.shape(mean)[0]
        N = tf.shape(mean)[1]
        z = tf.random.normal((S,B,N,1),dtype=tf.float64)
        
        I = self.jitter * tf.eye(N, dtype=tf.float64) #NN
        chol = tf.linalg.cholesky(var + I)  # BNN
        #samples = tf.expand_dims(mean,0) ## DEBUGGGGG removing sample and setting to the mean CHANGE BACK!!!!!!!!!!!! + tf.matmul(chol, z)#[:, :, :, 0]  # BSN1
        samples = tf.expand_dims(mean,0) + tf.matmul(chol, z)#[:, :, :, 0]  # BSN1
        return samples

    def full_conditional(self, Kmn, Kmm, Knn, f, q_sqrt, full_cov=True):

        f = tf.expand_dims(f,-1)
        q_sqrt= tf.expand_dims(q_sqrt,0)

        Lm = tf.linalg.cholesky(Kmm)

        N = tf.shape(Kmn)[-1]
        M = tf.shape(f)[0]

        # Compute the projection matrix A
        Lm = tf.broadcast_to(Lm, tf.shape(Lm))
        A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [..., M, N]

        # compute the covariance due to the conditioning
        fvar = Knn - tf.linalg.matmul(A, A, transpose_a=True)  # [..., N, N]

        # construct the conditional mean
        f_shape = [M, 1]
        f = tf.broadcast_to(f, f_shape)  # [..., M, R]
        fmean = tf.linalg.matmul(A, f, transpose_a=True)  # [..., N, R]

        L = tf.linalg.band_part(q_sqrt, -1, 0)  

        LTA = tf.linalg.matmul(L, A, transpose_a=True)  # [R, M, N]

        fvar = fvar + tf.linalg.matmul(LTA, LTA, transpose_a=True)  # [R, N, N]

        return fmean, fvar
    def full_conditional_lo(self, Kmn, Kmm, Knn, f, q_sqrt):

        f = tf.expand_dims(f,0)
        q_sqrt= tf.expand_dims(q_sqrt,0)
        Lm  = Kmm.cholesky()
        A = Lm.solve(Kmn)
        
        fmean = A.matvec(f,adjoint=True)
        B = A.matmul(A,adjoint=True)

        L = tf.linalg.LinearOperatorLowerTriangular(q_sqrt)
        LTA = L.matmul(A,adjoint=True)
        LTA = LTA.matmul(LTA,adjoint=True)

        fvar = Knn.to_dense() - B.to_dense() + LTA.to_dense()
        
        return tf.expand_dims(fmean,-1), fvar

    def marginal(self, Kmn, Kmm, Knn, f, q_sqrt):

        f = tf.expand_dims(f,-1)
        q_sqrt= tf.expand_dims(q_sqrt,0)

        Knn = tf.linalg.diag_part(Knn)
        Lm = tf.linalg.cholesky(Kmm)

        N = tf.shape(Kmn)[-1]
        M = tf.shape(f)[0]

        # Compute the projection matrix A
        Lm = tf.broadcast_to(Lm, tf.shape(Lm))
        A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [..., M, N]

        # compute the covariance due to the conditioning
        fvar = Knn - tf.reduce_sum(tf.square(A), -2)  # [..., N]

        # construct the conditional mean
        f_shape = [M, 1]
        f = tf.broadcast_to(f, f_shape)  # [..., M, R]
        fmean = tf.linalg.matmul(A, f, transpose_a=True)  # [..., N, R]

        L = tf.linalg.band_part(q_sqrt, -1, 0)  

        LTA = tf.linalg.matmul(L, A, transpose_a=True)  # [R, M, N]

        fvar = fvar + tf.reduce_sum(tf.square(LTA), -2)  # [R, N]
        fvar = tf.linalg.adjoint(fvar)  # [N, R]

        return fmean, fvar
    
        
    def non_stat_vel(self,T,lengthscales, var):
        
        """Non-stationary integrated Matern12 kernel"""
        stddev = tf.math.sqrt(var)
        sigma_ = 0.5*(stddev[...,:-1,0,None] + stddev[...,1:,0,None])
        len_ = 0.5*(lengthscales[...,:-1,0,None] + lengthscales[...,1:,0,None])

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
    
