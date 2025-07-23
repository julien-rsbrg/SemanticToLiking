import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, _check_length_scale

import scipy.optimize
from scipy.spatial.distance import cdist, pdist, squareform
import numpy as np
import torch


from src.models.generic_model import GenericModel
from src.utils import read_yaml,save_yaml

class MyRBF(RBF):
    """Radial basis function kernel (aka squared-exponential kernel).

    Minimal changes for entering the distance function as an attribute
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),dist_metric:str="sqeuclidean"):
        """
        Parameters
        ----------
        - dist_metric : (str)
            Check scipy documentation on pdist or cdist for the possible dist_metric values.
            Careful: if the kernel is anysotrope (length_scale is a vector of size > 1), the formula of of the kernel puts d(x/length_scale,y/length_scale) which is different from  d(x,y)/length_scale**2 in for most dist_metric (except sq Euclidian dist)
            Careful: if dist_metric is different from [sqeuclidean] (TODO make list longer), will compute d(x,y)/length_scale**2 instead of d(x/length_scale,y/length_scale) 
        """
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.dist_metric = dist_metric


    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        assert isinstance(self.length_scale,float) or (self.dist_metric in ["sqeuclidean"])            

        if Y is None:
            dists = pdist(X / length_scale, metric=self.dist_metric)
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            
            if not(self.dist_metric in ["sqeuclidean"]):
                dists = cdist(X, Y, metric=self.dist_metric ) /  (length_scale**2)
            else:
                dists = cdist(X / length_scale, Y / length_scale, metric=self.dist_metric)

            K = np.exp(-0.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale**2
                )
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

class MyRBFGaussianProcessRegressor(GenericModel):
    def __init__(self,
                 dim_out:int, 
                 lengthscale:float = 1.0,
                 length_scale_bounds:float = (1e-5,1e5),
                 dist_metric:str = "sqeuclidean"):
        self.dim_out = dim_out
        self.gpr = GaussianProcessRegressor(
            kernel=MyRBF(length_scale=lengthscale,
                         length_scale_bounds=length_scale_bounds,
                         dist_metric=dist_metric), 
                         random_state=0,
                         normalize_y=True)
        self.length_scale_bounds = length_scale_bounds
        self.dist_metric = dist_metric

    def get_n_free_params(self):
        return 1

    @property
    def requires_base(self):
        return True
    
    @property
    def lengthscale(self):
        return self.gpr.kernel.length_scale
    
    @lengthscale.setter
    def lengthscale(self,value:float|int):
        self.gpr.kernel.length_scale = value

    def predict(self, # reorder parameters
                node_attr:torch.Tensor,
                x_pred_mask:torch.Tensor,
                x_base_mask:torch.Tensor,
                y_base:torch.Tensor,
                **kwargs): # transform into node_attr = graph.x
        """Careful compared to other models, node_attr should be the representational values of x, not the information they carry for y
        """
        x_pred = node_attr[x_pred_mask].numpy()
        x_base = node_attr[x_base_mask].numpy()
        y_base = y_base.numpy()

        self.gpr.fit(X = x_base, y = y_base)

        y_pred = self.gpr.predict(X = x_pred)

        if len(y_pred.shape) == 1 and len(y_base.shape) == 2:
            y_pred = y_pred[...,np.newaxis] 

        return torch.Tensor(y_pred)

    def mae_error_function(self, 
                           y_true:torch.Tensor,
                           node_attr:torch.Tensor,
                           x_pred_mask:torch.Tensor,
                           x_base_mask:torch.Tensor,
                           y_base:torch.Tensor,
                           **kwargs):
        y_pred = self.predict(
                node_attr = node_attr,
                x_pred_mask = x_pred_mask,
                x_base_mask = x_base_mask,
                y_base = y_base)
        mae_loss = torch.nn.L1Loss()
        mae_error = mae_loss(y_pred, y_true)
        mae_error = mae_error.detach()
        return mae_error
    
    def mse_error_function(self, 
                           y_true:torch.Tensor,
                           node_attr:torch.Tensor,
                           x_pred_mask:torch.Tensor,
                           x_base_mask:torch.Tensor,
                           y_base:torch.Tensor,
                           **kwargs):
        y_pred = self.predict(
                node_attr = node_attr,
                x_pred_mask = x_pred_mask,
                x_base_mask = x_base_mask,
                y_base = y_base)
        mse_loss = torch.nn.MSELoss()
        mse_error = mse_loss(y_pred, y_true)
        mse_error = mse_error.detach()
        return mse_error
    
    def _store_params(self):
        self._stored_lengthscale = self.lengthscale 

    def _restore_params(self):
        assert hasattr(self,"_stored_lengthscale")

        self.lengthscale = self._stored_lengthscale
    
    def fit(self,
            dataset,
            val_dataset = None, 
            bounds = [(1e-5,1e5)], # need to give more values if bias
            method = "dual_annealing",
            **kwargs # use kwargs for optimization hyperparameters
            ):
        batch_graph = next(iter(dataset))
        is_validation_present = hasattr(batch_graph,"val_mask")

        assert len(bounds) == self.get_n_free_params(), (f"len(bounds) = {len(bounds)} != {self.get_n_free_params()} = self.get_n_free_params()")

        def obj_func(theta,
                     y_true, 
                     node_attr,
                     x_pred_mask, 
                     x_base_mask,
                     y_base):
            self.lengthscale = theta[0]

            # MSE as loss function because of its link to log-marginal-likelihood
            mse_loss = self.mse_error_function(
                y_true = y_true,
                node_attr = node_attr,
                x_pred_mask = x_pred_mask,
                x_base_mask = x_base_mask,
                y_base = y_base)
            return mse_loss.numpy() 

        history = {"lengthscale":[],"func_value":[],"val_mae":[],"val_mse":[]}

        for batch_graph in dataset:            
            if hasattr(batch_graph,"base_mask"):
                base_mask = batch_graph.base_mask
            else:
                base_mask = torch.ones(batch_graph.num_nodes).to(torch.bool)
                base_mask[batch_graph.train_mask] = False

                if is_validation_present:
                    base_mask[batch_graph.val_mask] = False
            
            self._store_params()

            if method == "dual_annealing":
                opt_res = scipy.optimize.dual_annealing(
                    lambda theta: obj_func(theta=theta,
                                        y_true = batch_graph.y[batch_graph.train_mask],
                                        node_attr = batch_graph.x,
                                        x_pred_mask = batch_graph.train_mask,
                                        x_base_mask = base_mask,
                                        y_base = batch_graph.y[base_mask]
                                        ), 
                    bounds=np.array(bounds)
                )
                params_opt, func_min = opt_res.x, opt_res.fun
                history["lengthscale"].append(float(params_opt[0]))
                history["func_value"].append(float(func_min))
            elif method == "brute":
                theta_opt, func_min, record_theta, record_func = scipy.optimize.brute(
                    func = lambda theta: obj_func(theta=theta,
                                        y_true = batch_graph.y[batch_graph.train_mask],
                                        node_attr = batch_graph.x,
                                        x_pred_mask = batch_graph.train_mask,
                                        x_base_mask = base_mask,
                                        y_base = batch_graph.y[base_mask]
                                        ), 
                    ranges=np.array(bounds),
                    Ns = kwargs["Ns"] if "Ns" in kwargs else 100,
                    full_output = True,
                    finish = None, # no polish, only raw values,
                    workers = 1 # TODO: improve
                )
                theta_opt, func_min, record_theta, record_func = float(theta_opt), float(func_min), [float(v) for v in record_theta], [float(v) for v in record_func]
                history["lengthscale"] += record_theta
                history["func_value"] += record_func
                history["val_mae"] += [np.nan for _ in range(len(record_func))]
                history["val_mse"] += [np.nan for _ in range(len(record_func))]

                history["lengthscale"] += [theta_opt]
                history["func_value"] += [func_min]
            else:
                raise NotImplementedError(f"Unknown optimization method: {method}")

            self._restore_params()

            

            if is_validation_present and batch_graph.val_mask.sum() > 0:
                init_lengthscale = self.lengthscale # no copy.copy because there is no "effet de bord" from float
                self.lengthscale = history["lengthscale"][-1]
                val_mae = self.mae_error_function(
                    y_true = batch_graph.y[batch_graph.val_mask],
                    node_attr = batch_graph.x,
                    x_pred_mask = batch_graph.val_mask,
                    x_base_mask = base_mask,
                    y_base = batch_graph.y[base_mask]
                )
                val_mse = self.mse_error_function(
                    y_true = batch_graph.y[batch_graph.val_mask],
                    x_pred_mask = batch_graph.val_mask,
                    x_base_mask = base_mask,
                    y_base = batch_graph.y[base_mask]
                )
                self.lengthscale = init_lengthscale
                history["val_mae"].append(float(val_mae))
                history["val_mse"].append(float(val_mse))
            
            else:

                history["val_mae"].append(np.nan)
                history["val_mse"].append(np.nan)
        
        # restore last best fit
        self.lengthscale = history["lengthscale"][-1]
        return history
    
    def get_dict_params(self):
        params = {
            "dim_out":self.dim_out,
            "lengthscale":self.lengthscale,
            "length_scale_bounds":self.length_scale_bounds,
            "dist_metric":self.dist_metric
        }        
        return params

    def get_config(self):
        """get configuration for the model"""
        config = {
            "name":"MyRBFGaussianProcessRegressor",
            "parameters":self.get_dict_params(),
            "n_free_params":self.get_n_free_params()
        }
        return config
    
    def save(self,dst_path:str):
        """save the model
        
        Parameters
        ----------
        dst_path : str
            Path to save the model. No extension.
        """
        save_yaml(data = self.get_config(),dst_path=dst_path+".yml")


    def load(self,src_path:str):
        """load the model"""
        config = read_yaml(src_path=src_path)
        self = MyRBFGaussianProcessRegressor(**config["parameters"])
        return self
    
    def reset_parameters(self):
        return super().reset_parameters()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch_geometric.data import Data

    def my_problem(X):
        return X[...,1]**2 + np.sin(2*np.pi*X[...,0])

    X = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
    X = np.stack(X).T
    y = my_problem(X)

    plt.imshow(y, cmap='hot', interpolation='nearest')
    plt.show()

    _X = X.reshape((X.shape[0]*X.shape[1],X.shape[-1]))
    print(_X.shape)

    _y = y.reshape((y.shape[0]*y.shape[1],1))
    print(_y.shape)

    n_samples_base = 200
    base_kept_ids = np.random.choice(np.arange(_X.shape[0]),size=n_samples_base,replace=False)
    _X_base = _X[base_kept_ids]
    _y_base = _y[base_kept_ids]

    n_samples_train = 1000
    train_kept_ids = np.random.choice(list(set(list(np.arange(_X.shape[0]))) - set(base_kept_ids)), size=n_samples_train, replace=False)
    _X_train = _X[train_kept_ids]
    _y_train = _y[train_kept_ids]

    new_X = np.concat([_X_base,_X_train],axis=0)
    new_y = np.concat([_y_base,_y_train],axis=0)

    base_mask = torch.zeros(new_X.shape[0]).to(bool)
    base_mask[:_X_base.shape[0]] = True
    train_mask = torch.zeros(new_X.shape[0]).to(bool)
    train_mask[_X_base.shape[0]:] = True
    complete_train_mask = train_mask.clone()

    graph = Data(
        x = torch.Tensor(new_X),
        y = torch.Tensor(new_y),
        x_names = ["x","y"],
        y_names = ["f(x,y)"],
        train_mask = train_mask,
        base_mask = base_mask,
        complete_train_mask = complete_train_mask
    )

    gpr = MyRBFGaussianProcessRegressor(dim_out=1,lengthscale=1,length_scale_bounds="fixed")
    y_pred = gpr.predict(
        node_attr=graph.x,
        x_pred_mask=torch.ones(graph.x.size(0)).to(bool),
        x_base_mask = graph.base_mask,
        y_base = graph.y[graph.base_mask]
    )

    plt.scatter(graph.x[graph.base_mask,0],graph.x[graph.base_mask,1],c=graph.y[graph.base_mask])
    plt.show()

    plt.scatter(graph.x[graph.train_mask,0],graph.x[graph.train_mask,1],c=y_pred[graph.train_mask])
    plt.show()

    ## Parameter Recovery
    graph.y = y_pred

    gpr = MyRBFGaussianProcessRegressor(dim_out=1,lengthscale=1,length_scale_bounds="fixed")
    
    start_time = time.time()
    history = gpr.fit(
        dataset=[graph],
        bounds=[(1e-5,1e5)],
        method="brute",
        Ns = 100,
        workers = 1 # doesn't work (yet)
    )
    fit_time_taken = time.time()-start_time
    print("fit time taken: {}h - {}m - {}s".format(fit_time_taken//(3600),((fit_time_taken%3600)//60),((fit_time_taken%3600)%60)))
    print("history:\n",history)

    print("gpr.lengthscale:",gpr.lengthscale)

    y_pred = gpr.predict(
        node_attr=graph.x,
        x_pred_mask=torch.ones(graph.x.size(0)).to(bool),
        x_base_mask = graph.base_mask,
        y_base = graph.y[graph.base_mask]
    )
    plt.scatter(graph.x[graph.train_mask,0],graph.x[graph.train_mask,1],c=y_pred[graph.train_mask])
    plt.show()
