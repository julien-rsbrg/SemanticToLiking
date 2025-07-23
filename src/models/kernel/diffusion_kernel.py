import copy

import numpy as np
import torch
import scipy.sparse

from geometric_kernels.spaces import Graph
from geometric_kernels.kernels import MaternGeometricKernel

import networkx as nx

from src.models.generic_model import GenericModel
from src.utils import read_yaml,save_yaml

def convert_to_adj_matrix(num_nodes:int,edge_index:np.ndarray,edge_weight:np.ndarray | None = None):
    if edge_weight is None:
        _edge_weight = np.ones(edge_index.shape[1])
    else:
        _edge_weight = np.copy(edge_weight)

    adj_matrix = scipy.sparse.coo_matrix((_edge_weight, (edge_index[0,:], edge_index[1,:])), shape=(num_nodes, num_nodes)).toarray()
    return adj_matrix

class MaternKernelModel(GenericModel):
    def __init__(self, 
                 dim_out:int, 
                 nu: float = np.inf, 
                 lengthscale: float = 1.0, 
                 use_bias: bool = False, 
                 **kwargs):
        assert dim_out > 0
        assert isinstance(nu,float), ("nu: "+str(nu)) 
        assert isinstance(lengthscale,float), ("lengthscale: "+str(nu)) 

        self.dim_out = dim_out
        self.nu = nu
        self.lengthscale = lengthscale

        # init_kernel before use
        self.kernel = None

        self.use_bias = use_bias

        if self.use_bias:
            raise NotImplementedError("This option is outdated")
            for k in kwargs:
                if k.split("_") =="bias":
                    bias_id = int(k.split("_")[1])
                    self._bias[bias_id] = kwargs[k]

    @property 
    def bias(self):
        return self._bias
    
    @bias.setter
    def bias(self,value:float|torch.Tensor):
        raise NotImplementedError("This option is outdated")
        self._use_bias = True
        if isinstance(value,float) or isinstance(value,int):
            self._bias = torch.empty(self.dim_out).fill_(value)
        else:
            self._bias = value

    @property 
    def use_bias(self):
        return self._use_bias
    
    @use_bias.setter
    def use_bias(self,value:bool):
        self._use_bias = value
        if value and not(hasattr(self,'_bias')):
            self._bias = torch.zeros(self.dim_out)

    @property
    def requires_base(self):
        return True
    
    def _init_kernel(self,
                    num_nodes:int,
                    edge_index:torch.Tensor,
                    edge_attr:torch.Tensor):
        assert len(edge_attr.size()) == 1 or (len(edge_attr.size())==2 and edge_attr.size(1) == 1), ("wrong size for edge_attr:", edge_attr.size())
        edge_attr = edge_attr.flatten()
        adj_matrix = convert_to_adj_matrix(num_nodes,edge_index.numpy(), edge_weight = edge_attr.numpy())
        graph = Graph(np.array(adj_matrix), normalize_laplacian=False)
        self.kernel = MaternGeometricKernel(graph)
        return self


    def predict(self, # reorder parameters
                edge_index:torch.Tensor,
                edge_attr:torch.Tensor, # rename in edge_attr and check dim(-1) == 1
                x_pred_mask:torch.Tensor,
                x_base_mask:torch.Tensor,
                y_base:torch.Tensor,
                num_nodes:int,
                **kwargs): # transform into node_attr = graph.x
        assert len(edge_attr.size()) == 1 or (len(edge_attr.size())==2 and edge_attr.size(1) == 1), ("wrong size for edge_attr:", edge_attr.size())
        edge_attr = edge_attr.flatten()
        self._init_kernel(num_nodes,edge_index,edge_attr)
        x_pred = torch.where(x_pred_mask)[0].numpy()[...,None]
        x_base = torch.where(x_base_mask)[0].numpy()[...,None]

        params = {
            "nu":np.array([self.nu]),
            "lengthscale":np.array([self.lengthscale])
        }

        kernel_mat = self.kernel.K(params, x_pred, x_base)
        kernel_mat_base = self.kernel.K(params, x_base, x_base)
        y_pred = (kernel_mat @ (scipy.linalg.inv(kernel_mat_base + 1e-10 * np.eye(y_base.size(dim=0))))) @ y_base.numpy()

        if self.use_bias:
            return torch.Tensor(y_pred) + self.bias
        else:
            return torch.Tensor(y_pred)

    def mae_error_function(self, 
                           y_true:torch.Tensor,
                           edge_index:torch.Tensor,
                           edge_attr:torch.Tensor, # rename in edge_attr and check dim(-1) == 1
                           x_pred_mask:torch.Tensor,
                           x_base_mask:torch.Tensor,
                           y_base:torch.Tensor,
                           num_nodes:int,
                           **kwargs):
        y_pred = self.predict(
                edge_index = edge_index,
                edge_attr = edge_attr,
                x_pred_mask = x_pred_mask,
                x_base_mask = x_base_mask,
                y_base = y_base,
                num_nodes = num_nodes)
        mae_loss = torch.nn.L1Loss()
        mae_error = mae_loss(y_pred, y_true)
        mae_error = mae_error.detach()
        return mae_error
    

    def mse_error_function(self, 
                           y_true:torch.Tensor,
                           edge_index:torch.Tensor,
                           edge_attr:torch.Tensor, # rename in edge_attr and check dim(-1) == 1
                           x_pred_mask:torch.Tensor,
                           x_base_mask:torch.Tensor,
                           y_base:torch.Tensor,
                           num_nodes:int,
                           **kwargs):
        y_pred = self.predict(
                edge_index = edge_index,
                edge_attr = edge_attr,
                x_pred_mask = x_pred_mask,
                x_base_mask = x_base_mask,
                y_base = y_base,
                num_nodes = num_nodes)
        mae_loss = torch.nn.MSELoss()
        mae_error = mae_loss(y_pred, y_true)
        mae_error = mae_error.detach()
        return mae_error



    def _store_params(self):
        self._init_lengthscale = self.lengthscale # no copy.copy because there is no "effet de bord" from float
        
        if self.use_bias:
            if isinstance(self.bias,torch.Tensor):
                self._init_bias = self.bias.clone()
            else:
                self._init_bias = self.bias # no copy.copy because there is no "effet de bord" from float
 


    def _restore_params(self):
        self.lengthscale = self._init_lengthscale # no copy.copy because there is no "effet de bord" from float
        
        if self.use_bias:
            if isinstance(self._init_bias,torch.Tensor):
                self.bias = self._init_bias.clone()
            else:
                self.bias = self._init_bias # no copy.copy because there is no "effet de bord" from float
        
    
    def fit(self,
            dataset,
            edge_weight_name:str,
            val_dataset = None, 
            bounds = [(1e-5,1e5)], # need to give more values if bias
            method = "dual_annealing",
            **kwargs # use kwargs for optimization hyperparameters
            ):
        batch_graph = next(iter(dataset))
        assert hasattr(batch_graph,"edge_attr_names"),batch_graph
        is_validation_present = hasattr(batch_graph,"val_mask")

        assert len(bounds) == self.get_n_free_params(), (f"len(bounds) = {len(bounds)} != {self.get_n_free_params()} = self.get_n_free_params()")

        def obj_func(theta,
                     y_true, 
                     edge_index,
                     edge_attr,
                     x_pred_mask, 
                     x_base_mask,
                     y_base,
                     num_nodes):
            self.lengthscale = theta[0]

            if self.use_bias:
                self.bias = torch.Tensor(theta[1:])

            # MSE as loss function because of its link to log-marginal-likelihood
            mse_loss = self.mse_error_function(
                y_true = y_true,
                edge_index = edge_index,
                edge_attr = edge_attr,
                x_pred_mask = x_pred_mask,
                x_base_mask = x_base_mask,
                y_base = y_base,
                num_nodes = num_nodes)
            return mse_loss.numpy() 

        history = {"lengthscale":[],"func_value":[],"val_mae":[],"val_mse":[]}
        if self.use_bias:
            history.update({f"bias_{i}_opt":[] for i in range(self.dim_out)})

        for batch_graph in dataset:            
            if hasattr(batch_graph,"base_mask"):
                base_mask = batch_graph.base_mask
            else:
                base_mask = torch.ones(batch_graph.num_nodes).to(torch.bool)
                base_mask[batch_graph.train_mask] = False

                if is_validation_present:
                    base_mask[batch_graph.val_mask] = False

            edge_weight_id = np.where(np.array(batch_graph.edge_attr_names) == edge_weight_name)[0][0]
            
            self._store_params()
            if method == "dual_annealing":
                opt_res = scipy.optimize.dual_annealing(
                    lambda theta: obj_func(theta=theta,
                                        y_true = batch_graph.y[batch_graph.train_mask],
                                        edge_index = batch_graph.edge_index,
                                        edge_attr = batch_graph.edge_attr[:,edge_weight_id],
                                        x_pred_mask = batch_graph.train_mask,
                                        x_base_mask = base_mask,
                                        y_base = batch_graph.y[base_mask],
                                        num_nodes = batch_graph.num_nodes), 
                    bounds=np.array(bounds)
                )
                params_opt, func_min = opt_res.x, opt_res.fun
                history["lengthscale"].append(float(params_opt[0]))
                history["func_value"].append(float(func_min))

            elif method == "brute":
                theta_opt, func_min, record_theta, record_func = scipy.optimize.brute(
                    func = lambda theta: obj_func(theta=theta,
                                        y_true = batch_graph.y[batch_graph.train_mask],
                                        edge_index = batch_graph.edge_index,
                                        edge_attr = batch_graph.edge_attr[:,edge_weight_id],
                                        x_pred_mask = batch_graph.train_mask,
                                        x_base_mask = base_mask,
                                        y_base = batch_graph.y[base_mask],
                                        num_nodes = batch_graph.num_nodes), 
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

            if self.use_bias:
                for i in range(1,len(params_opt)):
                    history[f"bias_{i-1}_opt"].append(float(params_opt[i]))

            if is_validation_present:
                init_lengthscale = self.lengthscale # no copy.copy because there is no "effet de bord" from float
                self.lengthscale = history["lengthscale"][-1]
                val_mae = self.mae_error_function(
                    y_true = batch_graph.y[batch_graph.val_mask],
                    edge_index = batch_graph.edge_index,
                    edge_attr = batch_graph.edge_attr[:,edge_weight_id],
                    x_pred_mask = batch_graph.val_mask,
                    x_base_mask = base_mask,
                    y_base = batch_graph.y[base_mask],
                    num_nodes = batch_graph.num_nodes
                )
                val_mse = self.mse_error_function(
                    y_true = batch_graph.y[batch_graph.val_mask],
                    edge_index = batch_graph.edge_index,
                    edge_attr = batch_graph.edge_attr[:,edge_weight_id],
                    x_pred_mask = batch_graph.val_mask,
                    x_base_mask = base_mask,
                    y_base = batch_graph.y[base_mask],
                    num_nodes = batch_graph.num_nodes
                )
                self.lengthscale = init_lengthscale
                history["val_mae"].append(float(val_mae))
                history["val_mse"].append(float(val_mse))
            
            else:

                history["val_mae"].append(np.nan)
                history["val_mse"].append(np.nan)
        
        # restore last best fit
        self.lengthscale = history["lengthscale"][-1]
        if self.use_bias:
            if self.dim_out == 1:
                self.bias = history["bias_0_opt"][-1]
            else:
                self.bias = torch.Tensor([history[f"bias_{i}_opt"][-1] for i in range(self.dim_out)])
        return history


    def get_dict_params(self):
        params = {
            "dim_out":self.dim_out, 
            "nu":self.nu,
            "lengthscale":self.lengthscale,
            "use_bias":self.use_bias
        }

        if self.use_bias:
            new_params = {}
            for i in range(len(self.bias)):
                new_params[f"bias_{i}"] = float(self.bias[i])
        
            params.update(new_params)
        
        return params
    
    def get_n_free_params(self):
        n_free_params = 1 # lengthscale
        if self.use_bias:
            n_free_params += len(self.bias.flatten())
        return n_free_params


    def get_config(self):
        """get configuration for the model"""
        config = {
            "name":"MaternKernelModel",
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
        self = MaternKernelModel(**config["parameters"])
        return self
    

    def reset_parameters(self):
        return super().reset_parameters()
    



if __name__ == "__main__":
    from torch_geometric.data import Data
    import matplotlib.pyplot as plt
    import torch_geometric.transforms as T
    from src.visualization.display_graph import convert_torch_to_networkx_graph

    matern_model = MaternKernelModel(dim_out=1, nu = np.inf)

    graph = Data(
            x = torch.Tensor([[0,0],
                            [1,0],
                            [2,0],
                            [2,1]]),
            x_names = ["values","experience"],
            y = torch.Tensor([[0.80153107],
                            [0.3940955 ],
                            [1.        ],
                            [1.]]),
            y_names = ["values"],
            edge_index = torch.Tensor([[0,1,2,2],
                                [1,2,3,0]]).to(torch.int64),
            edge_attr = torch.Tensor([[1],
                                    [0.5],
                                    [1],
                                    [1]]),
            edge_attr_names = ["weight_name"],
            train_mask = torch.Tensor([True,True,False,False]).to(bool),
            val_mask = torch.Tensor([False,False,False,False]).to(bool),
            base_mask = torch.Tensor([False,False,False,True]).to(bool),
            complete_train_mask = torch.Tensor([True,True,False,False]).to(bool)
        )

    transform = T.Compose([T.ToUndirected(reduce="mean")])
    graph = transform(graph)

    # test fit and predict
    for lengthscale in np.linspace(1,5,5):
        matern_model.lengthscale = lengthscale
        matern_model.bias = 15
        print("matern_model.lengthscale",matern_model.lengthscale)
        print("matern_model.use_bias",matern_model.use_bias)
        print("matern_model.bias",matern_model.bias)

        # nx.draw_spring(convert_torch_to_networkx_graph(graph), with_labels=True)
        # plt.show()

        y_pred = matern_model.predict(
            edge_index = graph.edge_index,
            edge_attr = graph.edge_attr[:,0],
            x_pred_mask = torch.Tensor([True,True,True,True]),
            x_base_mask = torch.Tensor([False,False,False,True]),
            y_base = torch.Tensor([[1]]),
            num_nodes = graph.num_nodes
        )
        print("y_pred\n",y_pred)
        graph.y = y_pred

        print("graph.y\n",graph.y)

        matern_model.lengthscale = np.random.randn()*0.5 + 1
        matern_model.bias = np.random.randint(1,50)
        history = matern_model.fit(
            dataset=[graph],
            edge_weight_name="weight_name",
            bounds = [(1e-1,1e1),(-20,20)]
        )
        print(history)

        matern_model.lengthscale = history["lengthscale"][-1]
        y_pred = matern_model.predict(
            edge_index = graph.edge_index,
            edge_attr = graph.edge_attr[:,0],
            x_pred_mask = torch.Tensor([True,True,True,True]),
            x_base_mask = torch.Tensor([False,False,False,True]),
            y_base = torch.Tensor([[1]]),
            num_nodes = graph.num_nodes
        )
        print("y_pred\n",y_pred)
        break
    
    ##########################################
    from src.processing.preprocessing import PreprocessingPipeline, NoValidationHandler,MaskThreshold
    from src.models.model_pipeline import ModelPipeline

    model_name = "MaternKernelInf"
    
    supplementary_config = {
        "model_name":model_name,
        "sim_used":"original",
        "model_fit_params":{}}
    
    validation_handler = NoValidationHandler()
    model_pipeline = ModelPipeline(
        preprocessing_pipeline = PreprocessingPipeline(
            complete_train_mask_selector=MaskThreshold(feature_name="experience",threshold=0,mode="lower"),
            transformators = [],
            validation_handler = validation_handler,
            base_mask_selector=MaskThreshold(feature_name="experience",threshold=0,mode="strict_upper"),
        ),
        model = MaternKernelModel(dim_out=1, lengthscale=1e-3),
        dst_folder_path = "replication_old_results/test_pipeline"
    )  

    supplementary_config["model_fit_params"] = dict(
        complete_train_mask = graph.complete_train_mask,
        edge_weight_name = "weight_name",
        bounds = [(1e-2,1e2)]
    )

            
    _supplementary_config = copy.copy(supplementary_config) # for saving
    _supplementary_config["model_fit_params"] = dict(
            epochs = 10000,
            report_epoch_steps = 1
        )

    model_pipeline.save_config(supplementary_config = _supplementary_config)

    complete_train_mask = graph.complete_train_mask
    train_val_sets = model_pipeline.preprocessing_pipeline.validation_handler.apply(complete_train_mask)

    graphs_dataset = []
    for i in range(len(train_val_sets)):
        data_graph = graph.clone()
        data_graph.train_mask = torch.Tensor(train_val_sets[i][0])
        data_graph.val_mask = torch.Tensor(train_val_sets[i][1])
        graphs_dataset.append(data_graph)

    pred_values = model_pipeline.predict(
        graph=graph,
        data_state="preprocessed"
    )
    print("pred_values\n",pred_values)

    model_pipeline.run_models(graphs_dataset=graphs_dataset, **supplementary_config["model_fit_params"])

    pred_values = model_pipeline.predict(
        graph=graph,
        data_state="preprocessed"
    )
    print("pred_values\n",pred_values)
    
