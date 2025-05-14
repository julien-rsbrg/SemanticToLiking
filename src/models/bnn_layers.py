import copy
import numpy as np

import torch
from torch.autograd import Variable

from src.models.utils.common import log_norm


class VIModule(torch.nn.Module):
    """
    A mixin class to attach loss functions to layer. This is usefull when doing variational inference with deep learning.

    inspired from: Jospin et al. 2022, 10.1109/MCI.2022.3155327
    """
    pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._internal_losses = []
        self.loss_scale_factor = 1

    def add_loss(self, func):
        self._internal_losses.append(func)

    def eval_losses(self):
        t_loss = 0

        for l in self._internal_losses:
            new_loss = l()
            if not(torch.any(torch.isnan(new_loss))):
                t_loss = t_loss + new_loss

        return t_loss

    def eval_all_losses(self):
        t_loss = self.eval_losses()

        for m in self.children():
            if isinstance(m, VIModule):
                new_term = m.eval_all_losses()*self.loss_scale_factor
                t_loss = t_loss + new_term
        return t_loss


class MeanFieldGaussianFeedForward(VIModule):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_weights_m,
                 prior_weights_s,
                 likelihood_s=5.5,
                 n_latent: int = 100,
                 has_bias: bool = False,
                 prior_bias_m: float = 0.0,
                 prior_bias_s: float = 1.0,
                 device=torch.device("cpu")):
        super(MeanFieldGaussianFeedForward, self).__init__()
        self.n_latent = n_latent  # Number of latent samples
        self.softplus = torch.nn.Softplus()
        self.in_features = in_features
        self.out_features = out_features

        self.device = device

        # The parameters we adjust during training.
        self.weights_m = torch.nn.Parameter(torch.randn(
            in_features, out_features,device=self.device), requires_grad=True)
        self.weights_s = torch.nn.Parameter(torch.randn(
            in_features, out_features,device=self.device), requires_grad=True)

        # create holders for prior mean and std, and likelihood std.
        self.prior_weights_m = Variable(torch.randn(
            in_features, out_features,device=self.device), requires_grad=False)
        self.prior_weights_s = Variable(torch.randn(
            in_features, out_features,device=self.device), requires_grad=False)
        self.likelihood_s = Variable(torch.FloatTensor(
            (1)), requires_grad=False).to(self.device)

        # Set the prior and likelihood moments.
        self.prior_weights_m.data.fill_(prior_weights_m)
        self.prior_weights_s.data.fill_(prior_weights_s)
        self.likelihood_s.data.fill_(likelihood_s)

        self.add_loss(self.compute_internal_KL_div_weights)

        # Bias
        self.has_bias = has_bias

        if has_bias:
            self.bias_m = torch.nn.Parameter(torch.randn(
                out_features), requires_grad=True, device=self.device)
            self.bias_s = torch.nn.Parameter(torch.randn(
                out_features), requires_grad=True, device=self.device)

            # create holders
            self.prior_bias_m = Variable(torch.randn(
                out_features), requires_grad=False, device=self.device)
            self.prior_bias_s = Variable(torch.randn(
                out_features), requires_grad=False, device=self.device)

            # Set the prior moments.
            self.prior_bias_m.data.fill_(prior_bias_m)
            self.prior_weights_s.data.fill_(prior_bias_s)

            self.add_loss(self.compute_internal_KL_div_bias)
        
        # Register parameters
        parameters = [self.weights_m, self.weights_s]
        if has_bias:
            parameters += [self.bias_m,self.bias_s]

        self.params = torch.nn.ParameterList(parameters)


    def __repr__(self):
        return "MeanFieldGaussianFeedForward(\nin_features:{},\nout_features:{},\nn_latent:{},\nsoftplus:{},\ndevice:{})".format(
            self.in_features, self.out_features, self.n_latent, self.softplus, self.device)

    def sample_weights(self):
        eps = np.random.normal(
            size=(self.n_latent, self.in_features, self.out_features))
        eps = Variable(torch.FloatTensor(eps)).to(self.device)

        self.w_noise_weights = eps
        self.sampled_weights = (
            eps*self.softplus(self.weights_s)).add(self.weights_m)

    def sample_biases(self):
        eps = np.random.normal(size=(self.n_latent, self.out_features))
        eps = Variable(torch.FloatTensor(eps)).to(self.device)

        self.w_noise_biases = eps
        self.sampled_biases = (eps*self.softplus(self.bias_s)).add(self.bias_m)

    def sample_all_parameters(self):
        self.sample_weights()

        if self.has_bias:
            self.sample_biases()

    def forward(self, x, retrieve_latent = False):
        self.sample_all_parameters()
        preds = torch.einsum("ij,kjl->kil", x, self.sampled_weights)
        if retrieve_latent:
            return preds
        else:
            return preds.mean(0)

    def compute_likelihood(self, preds, labels):
        likelihood = torch.mean(
            torch.sum(log_norm(labels, preds, self.likelihood_s), 0))
        return likelihood

    def compute_internal_KL_div_weights(self):
        q_likelihood = log_norm(self.sampled_weights,
                                self.weights_m, self.softplus(self.weights_s))
        q_likelihood = torch.mean(q_likelihood)

        prior = log_norm(self.sampled_weights, self.prior_weights_m,
                         self.softplus(self.prior_weights_s))
        prior = torch.mean(prior)

        return q_likelihood - prior

    def compute_internal_KL_div_bias(self):
        q_likelihood = log_norm(self.sampled_biases,
                                self.bias_m, self.softplus(self.bias_s))
        q_likelihood = torch.mean(q_likelihood)

        prior = log_norm(self.sampled_biases,
                         self.prior_bias_m, self.prior_bias_s)
        prior = torch.mean(prior)

        return q_likelihood - prior
