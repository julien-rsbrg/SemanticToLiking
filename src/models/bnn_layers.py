import copy
import numpy as np

import torch
from torch.autograd import Variable


## utils ##

def log_norm(x, mu, std):
    """Compute the log pdf of x,
    under a normal distribution with mean mu and standard deviation std."""
    
    return -0.5 * torch.log(2*np.pi*std**2) - (0.5 * (1/(std**2))* (x-mu)**2)



## modules ##

class VIModule(torch.nn.Module) :
	"""
	A mixin class to attach loss functions to layer. This is usefull when doing variational inference with deep learning.
	
	inspired from: Jospin et al. 2022, 10.1109/MCI.2022.3155327
	"""
	
	def __init__(self, *args, **kwargs) :
		super().__init__(*args, **kwargs)
		
		self._internal_losses = []
		self.loss_scale_factor = 1
		
	def add_loss(self, func) :
		self._internal_losses.append(func)
		
	def eval_losses(self) :
		t_loss = 0
		
		for l in self._internal_losses :
			t_loss = t_loss + l()
			
		return t_loss
	
	def eval_all_losses(self) :
		
		t_loss = self.eval_losses()*self.loss_scale_factor
		
        # add the losses that are contained in this module's submodules (if they are VIModules)
		for m in self.children() :
			if isinstance(m, VIModule) :
				t_loss = t_loss + m.eval_all_losses()*self.loss_scale_factor
				
		return t_loss
      


class MeanFieldGaussianFeedForward(VIModule):
    def __init__(self,
                 in_features,
                 out_features,
                 n_latent:int = 100,
                 has_bias:bool = False):
        super(MeanFieldGaussianFeedForward, self).__init__()
        self.n_latent = n_latent # Number of latent samples
        self.softplus = torch.nn.Softplus()
        self.in_features = in_features
        self.out_features = out_features
        
        #The parameters we adjust during training.
        self.weights_m = torch.nn.Parameter(torch.randn(in_features,out_features), requires_grad=True)
        self.weights_s = torch.nn.Parameter(torch.randn(in_features,out_features), requires_grad=True)
        
        #create holders for prior mean and std, and likelihood std.
        self.prior_weights_m = Variable(torch.randn(in_features,out_features), requires_grad=False)
        self.prior_weights_s = Variable(torch.randn(in_features,out_features), requires_grad=False)
        self.likelihood_s = Variable(torch.FloatTensor((1)), requires_grad=False)
        
        #Set the prior and likelihood moments.
        self.prior_weights_s.data.fill_(1.0)
        self.prior_weights_m.data.fill_(0.9)
        self.likelihood_s.data.fill_(5.5)

        self.add_loss(self.compute_internal_KL_div_weights)

        # Bias
        self.has_bias = has_bias

        if has_bias:
            self.bias_m = torch.nn.Parameter(torch.randn(out_features), requires_grad=True)
            self.bias_s = torch.nn.Parameter(torch.randn(out_features), requires_grad=True)

            #create holders
            self.prior_bias_m = Variable(torch.randn(out_features), requires_grad=False)
            self.prior_bias_s = Variable(torch.randn(out_features), requires_grad=False)
            
            #Set the prior moments.
            self.prior_bias_m.data.fill_(0.0)
            self.prior_weights_s.data.fill_(1.0)

            self.add_loss(self.compute_internal_KL_div_bias)
     
    
    def sample_weights(self):
        eps = np.random.normal(size=(self.n_latent,self.in_features,self.out_features))
        eps = Variable(torch.FloatTensor(eps))

        self.w_noise_weights = eps
        self.sampled_weights = (eps*self.softplus(self.weights_s)).add(self.weights_m) 
    

    def sample_biases(self):
        eps = np.random.normal(size=(self.n_latent,self.out_features))
        eps = Variable(torch.FloatTensor(eps))

        self.w_noise_biases = eps
        self.sampled_biases = (eps*self.softplus(self.bias_s)).add(self.bias_m) 
    

    def sample_all_parameters(self):
        self.sample_weights()

        if self.has_bias:
            self.sample_biases()

    
    def forward(self,x):
        self.sample_all_parameters()
        preds = torch.einsum("ij,kjl->kil",x,self.sampled_weights)
        return preds

    
    def compute_likelihood(self,preds,labels):
        likelihood = torch.mean(torch.sum(log_norm(labels, preds, self.likelihood_s), 0))
        return likelihood
    

    def compute_internal_KL_div_weights(self):
        q_likelihood = log_norm(self.sampled_weights, self.weights_m, self.softplus(self.weights_s))
        q_likelihood = torch.mean(q_likelihood)

        prior = log_norm(self.sampled_weights, self.prior_weights_m, self.softplus(self.prior_weights_s))
        prior = torch.mean(prior)

        return q_likelihood - prior
    

    def compute_internal_KL_div_bias(self):
        q_likelihood = log_norm(self.sampled_biases, self.bias_m, self.softplus(self.bias_s))
        q_likelihood = torch.mean(q_likelihood)

        prior = log_norm(self.sampled_biases, self.prior_bias_m, self.prior_bias_s)
        prior = torch.mean(prior)

        return q_likelihood - prior