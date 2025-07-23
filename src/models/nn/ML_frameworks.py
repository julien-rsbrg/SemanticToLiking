import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time

import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch import Tensor
import torch_geometric.nn
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils.sparse import set_sparse_value
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
    to_dense_adj
)
from torch_geometric.nn.dense.linear import Linear
from torch.autograd import Variable

from src.models.generic_model import GenericModel

import src.models.utils.callbacks as callbacks
import src.models.utils.common as common_utils
import src.models.utils.weight_constrainers as weight_constrainers


class GNNFramework(GenericModel):
    # TODO : Update **kwargs predict everywhere it applies (ex in loss_function)

    # Class initialization params
    def __init__(self, 
                 update_node_module, 
                 device):
        self.device = device
        self.update_node_module = update_node_module.to(self.device)


    def predict(self, node_attr:Tensor, edge_index:Tensor, edge_attr:Tensor|None = None, **kwargs):
        node_attr = node_attr.to(self.device)
        edge_index = edge_index.to(self.device)
        if not(edge_attr is None):
            edge_attr = edge_attr.to(self.device)

        model_out = self.update_node_module(node_attr, edge_index, edge_attr, **kwargs)
        return model_out
    
    @property
    def requires_base(self):
        return False

    # Compute the loss function   
    def loss_function(self, node_attr, edge_index, edge_attr, labels, mask, **kwargs):
        node_attr = node_attr.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device) if not(edge_attr is None) else None

        preds = self.predict(node_attr, edge_index, edge_attr, **kwargs)

        loss_fn = torch.nn.MSELoss()

        loss = loss_fn(preds[mask], labels[mask])
        return loss

    # Train step
    def configure_optimizer(self, lr=1e-3):
        # TODO: optimizer agnostic function, give the name of the optimizer in arg
        # TODO: should not be here actually... This class is for learning itself. The value of this function is to set an example. 
         
        optimizer = torch.optim.Adam(self.update_node_module.parameters(), lr=lr) # -> loss 0.7
        # optimizer = torch.optim.SGD(self.update_node_module.parameters(), lr=lr) # -> loss 0.7

        return optimizer

    def configure_scheduler(self, optimizer, start_factor, end_factor, total_iters):
        # TODO: should not be here actually... This class is for learning itself. The value of this function is to set an example.

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer, gamma)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
        return scheduler

    def configure_weight_constrainer(self,name,min_value = -torch.inf,max_value = torch.inf):
        # TODO: should not be here actually... This class is for learning itself. The value of this function is to set an example.
        
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer, gamma)
        if name == "clipper":
            weight_constrainer = weight_constrainers.WeightClipper(
                min_value=min_value, 
                max_value=max_value)
        elif name == "sigmoid":
            weight_constrainer = weight_constrainers.WeightSigmoid(
                min_value=min_value, 
                max_value=max_value)
        else:
            raise RuntimeError(f"Weight constrainer {name} not supported")

        return weight_constrainer
    

    def train_step(self, node_attr, edge_index, edge_attr, labels, train_mask, optimizer, weight_constrainer = None, l1_reg = 0, l2_reg = 0,**kwargs):
        self.update_node_module.train(True)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Compute the loss and its gradients
        # mask applied after so that neighbors are taken into account
        loss = self.loss_function(node_attr, edge_index, edge_attr, labels, train_mask,**kwargs)
        
        if l1_reg > 0:
            l1_norm = sum(p.abs().sum() for p in self.update_node_module.parameters())
            loss += l1_reg * l1_norm
        if l2_reg >0:
            l2_norm = sum(p.pow(2).sum() for p in self.update_node_module.parameters())
            loss += l2_reg * l2_norm

        loss.backward()

        # Adjust learning weights
        optimizer.step()
        loss = loss.detach()

        # Adjust parameters if strict restriction exists
        if not(weight_constrainer is None):
            weight_constrainer(self.update_node_module)

        self.update_node_module.eval()
        return loss

    def mae_error_function(self, node_attr, edge_index, edge_attr, labels, mask,**kwargs):
        # mask applied after so that neighbors are taken into account
        preds = self.predict(node_attr, edge_index, edge_attr,**kwargs)[mask]
        mae_loss = torch.nn.L1Loss()
        mae_error = mae_loss(preds, labels[mask])
        mae_error = mae_error.detach()
        return mae_error

    def fit(self, 
            dataset, 
            epochs, 
            report_epoch_steps, 
            optimizer, 
            scheduler,
            weight_constrainer=None,
            early_stopping_monitor="val_mae", 
            patience=torch.inf, 
            min_delta=1e-4,
            l1_reg = 0,
            l2_reg = 0, 
            verbose=False,
            **kwargs):
        """Training loop."""

        print("== start training ==")
        # Initialize the optimizer.
        history = {
            "epoch":[],
            "train_loss": [], 
            "train_mae": [],
            "val_mae": []
            }
        
        batch_graph = next(iter(dataset))
        is_validation_present = hasattr(batch_graph,"val_mask")
        
        assert early_stopping_monitor in history, f"{early_stopping_monitor} not a metric followed by the model"

        early_stopping_cb = callbacks.EarlyStopping(
            early_stopping_monitor, patience, min_delta=min_delta, retrieve_model=False)

        for epoch in range(epochs):
            epoch_history = {k: [] for k in history.keys()}
            epoch_start_time = time.time()

            # training
            for batch_i, batch_graph in enumerate(dataset):
                batch_node_attr = batch_graph.x.to(self.device)
                batch_edge_index = batch_graph.edge_index.to(self.device)
                batch_edge_attr = batch_graph.edge_attr.to(self.device) if not(batch_graph.edge_attr is None) else None
                batch_labels = batch_graph.y.to(self.device)
                batch_train_mask = batch_graph.train_mask.to(self.device)
                if is_validation_present:
                    batch_val_mask = batch_graph.val_mask.to(self.device)

                train_loss = self.train_step(
                    batch_node_attr, 
                    batch_edge_index, 
                    batch_edge_attr, 
                    batch_labels, 
                    batch_train_mask, 
                    optimizer,
                    weight_constrainer=weight_constrainer,
                    l1_reg=l1_reg,
                    l2_reg=l2_reg,
                    **kwargs)

                with torch.no_grad():
                    train_mae = self.mae_error_function(
                        batch_node_attr, 
                        batch_edge_index, 
                        batch_edge_attr, 
                        batch_labels, 
                        batch_train_mask,
                        **kwargs)
                    
                    if is_validation_present:
                        val_mae = self.mae_error_function(
                            batch_node_attr, 
                            batch_edge_index, 
                            batch_edge_attr, 
                            batch_labels, 
                            batch_val_mask,
                            **kwargs)

                if verbose:
                    txt = "epoch: {epoch:.0f}/{epochs:.0f},\n batch_i: {batch_i:.0f}/{n_batch:.0f},\n batch_size: {batch_size:.0f},\n train_loss: {train_loss:.4f},\n train_mae: {train_mae:.4f},\n val_mae: {val_mae:4f}\n"
                    
                    txt.format(epoch=epoch+1,
                               epochs=epochs,
                               batch_i=batch_i+1,
                               n_batch=len(dataset),
                               batch_size=batch_train_mask.sum(),
                               train_loss=train_loss.cpu().numpy(),
                               train_mae=train_mae.cpu().numpy(),
                               val_mae=val_mae.cpu().numpy() if is_validation_present else torch.inf)
                    
                    print(txt, flush=True)
                
                epoch_history["train_loss"].append(
                    (batch_train_mask.sum(), train_loss))
                epoch_history["train_mae"].append(
                    (batch_train_mask.sum(), train_mae))
                if is_validation_present:
                    epoch_history["val_mae"].append(
                        (batch_val_mask.sum(), val_mae))
                else:
                    epoch_history["val_mae"].append((1,torch.inf))
            

            epoch_end_time = time.time()
            epoch_time_duration = epoch_end_time - epoch_start_time

            train_loss = sum([event[0]*event[1]
                              for event in epoch_history["train_loss"]])/sum([event[0] for event in epoch_history["train_loss"]])
            train_mae = sum([event[0]*event[1]
                             for event in epoch_history["train_mae"]])/sum([event[0] for event in epoch_history["train_mae"]])
            val_mae = sum([event[0]*event[1]
                           for event in epoch_history["val_mae"]])/sum([event[0] for event in epoch_history["val_mae"]])
            
            history["epoch"].append(epoch)
            history['train_loss'].append(train_loss.cpu())
            history["train_mae"].append(train_mae.cpu())
            history["val_mae"].append(val_mae.cpu())

            if (epoch+1) % report_epoch_steps == 0:
                txt = "epoch: {epoch:.0f}/{epochs:.0f},\n train_loss: {train_loss:.4f},\n train_mae: {train_mae:.4f},\n val_mae: {val_mae:.4f},\n epoch_time_duration: {epoch_time_duration:.4f}\n"
                print(txt.format(epoch=epoch+1,
                                 epochs=epochs,
                                 train_loss=history['train_loss'][-1],
                                 train_mae=history['train_mae'][-1],
                                 val_mae=history['val_mae'][-1],
                                 epoch_time_duration=epoch_time_duration),
                                 flush=True)

            # change learning rate
            scheduler.step()

            # callbacks
            early_stop = early_stopping_cb.on_epoch_end(
                history, self.update_node_module)
            
            if early_stop:
                print("early stopping activated")
                break

        self.update_node_module.train(False)

        if early_stopping_cb.retrieve_model and early_stopping_cb.model_copy is not None:
            self.update_node_module = early_stopping_cb.model_copy

        print('== end training ==\n')

        return history
    
    def save(self,dst_path:str):
        _dst_path = os.path.splitext(dst_path)[0] 
        _dst_path = _dst_path+".pt"
        torch.save(self.update_node_module.state_dict(), _dst_path)

    def load(self,src_path:str):
        self.update_node_module.load_state_dict(torch.load(src_path,map_location=self.device))
        self.update_node_module.eval()
        return self

    def reset_parameters(self):
        self.update_node_module.reset_parameters()
    
    def get_config(self):
        config = {"device":self.device.type,
                  "update_node_module":self.update_node_module.get_config(),
                  "n_free_params":self.update_node_module.n_free_params}
        return config

    def get_dict_params(self):
        return self.update_node_module.get_dict_parameters()


class BGNNFramework(GNNFramework):
    """
    TO UPDATE 15-05
    """
    def __init__(self, 
                 update_node_module, 
                 device, 
                 likelihood_s:float = 0.5):
        self.device = device
        self.update_node_module = update_node_module.to(self.device)
        self.update_node_module.device = self.device
        
        self.likelihood_s = Variable(torch.FloatTensor((1)), requires_grad=False).to(self.device)
        self.likelihood_s.data.fill_(likelihood_s)


    def predict(self, node_attr, edge_index, edge_attr, **kwargs):
        node_attr = node_attr.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        model_out = self.update_node_module(node_attr, edge_index, edge_attr, **kwargs)
        return model_out
    

    def loss_function(self, node_attr, edge_index, edge_attr, labels, mask):
        node_attr = node_attr.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        preds = self.predict(node_attr, edge_index, edge_attr)

        loss = 0

        likelihood = torch.mean(torch.sum(common_utils.log_norm(labels, preds, self.likelihood_s), 0))
        print("likelihood:",likelihood)
        loss -= likelihood

        print("loss:",loss)

        #loss += self.update_node_module.eval_all_losses()
        print("loss:",loss)

        return loss











class BGNN2LevelsFramework:
    """
    TO UPDATE 15-05
    """

    # Class initialization params
    def __init__(self, 
                 update_node_module, 
                 device, 
                 mask_node_fn:Optional[Callable] = None, 
                 mask_node_attr:Optional[torch.Tensor] = None):
        self.device = device
        self.update_node_module = update_node_module.to(self.device)
        self.update_node_module.device = device
        
        self.mask_node_fn = mask_node_fn
        self.mask_node_attr = mask_node_attr


    # Compute the solution of the PDE on the points (x,y)
    def predict(self, node_attr, edge_index, edge_attr, group_id, **kwargs):
        node_attr = node_attr.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        if not((self.mask_node_fn is None) or (self.mask_node_attr is None)):
            _node_attr = common_utils.replace_by_value(
                node_attr,
                mask_samples=self.mask_node_fn(node_attr),
                mask_attr=self.mask_node_attr
            )
        else:
            _node_attr = node_attr.clone()

        model_out = self.update_node_module(_node_attr, edge_index, edge_attr, group_id, **kwargs)
        return model_out

    # Compute the loss function   
    def loss_function(self, node_attr, edge_index, edge_attr, labels, group_id):
        node_attr = node_attr.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        elbo = self.update_node_module.compute_elbo(node_attr, edge_index, edge_attr, labels, group_id) ###
        loss = - elbo
        return loss

    # Train step
    def configure_optimizer(self, lr=1e-3):
        optimizer = torch.optim.Adam(
            self.update_node_module.parameters(), lr=lr)
        return optimizer

    def configure_scheduler(self, optimizer, start_factor, end_factor, total_iters):
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer, gamma)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
        return scheduler

    def train_step(self, node_attr, edge_index, edge_attr, labels, group_id, optimizer):
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Compute the loss and its gradients
        # mask applied after so that neighbors are taken into account
        loss = self.loss_function(node_attr, edge_index, edge_attr, labels, group_id)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        loss = loss.detach()
        return loss

    def mae_error_function(self, node_attr, edge_index, edge_attr, labels, mask):
        # mask applied after so that neighbors are taken into account
        preds = self.predict(node_attr, edge_index, edge_attr)[mask]
        mae_loss = torch.nn.L1Loss()
        mae_error = mae_loss(preds, labels[mask])
        mae_error = mae_error.detach()
        return mae_error

    def fit(self, 
            dataset, 
            epochs, 
            report_epoch_steps, 
            optimizer, 
            scheduler, 
            val_dataset=None,
            early_stopping_monitor="val_mae", 
            patience=torch.inf, 
            min_delta=1e-5, 
            verbose=False):
        """Training loop."""

        print("== start training ==")
        # Initialize the optimizer.
        history = {
            "epoch":[],
            "train_loss": [], 
            "train_mae": [],
            #"val_loss": [], #TODO
            #"val_mae": []   #TODO
            }
        early_stopping_cb = callbacks.EarlyStopping(
            early_stopping_monitor, patience, min_delta=min_delta)

        for epoch in range(epochs):
            epoch_history = {k: [] for k in history.keys()}
            epoch_start_time = time.time()

            # training
            self.update_node_module.train(True)
            for batch_i, batch_graph in enumerate(dataset):
                batch_node_attr = batch_graph.x.to(self.device)
                batch_edge_index = batch_graph.edge_index.to(self.device)
                batch_edge_attr = batch_graph.edge_attr.to(self.device)
                batch_labels = batch_graph.y.to(self.device)
                batch_train_mask = batch_graph.train_mask.to(self.device)

                train_loss = self.train_step(
                    batch_node_attr, 
                    batch_edge_index, 
                    batch_edge_attr, 
                    batch_labels, 
                    batch_train_mask, 
                    optimizer)

                train_mae = self.mae_error_function(
                    batch_node_attr, 
                    batch_edge_index, 
                    batch_edge_attr, 
                    batch_labels, 
                    batch_train_mask)

                if verbose:
                    txt = "epoch: {epoch:.0f}/{epochs:.0f},\n batch_i: {batch_i:.0f}/{n_batch:.0f},\n batch_size: {batch_size:.0f},\n train_loss: {train_loss:.4f},\n train_mae: {train_mae:.4f}\n"
                    print(txt.format(epoch=epoch+1,
                                     epochs=epochs,
                                     batch_i=batch_i+1,
                                     n_batch=len(dataset),
                                     batch_size=batch_train_mask.sum(),
                                     train_loss=train_loss.cpu().numpy(),
                                     train_mae=train_mae.cpu().numpy()), flush=True)
                epoch_history["train_loss"].append(
                    (batch_train_mask.sum(), train_loss))
                epoch_history["train_mae"].append(
                    (batch_train_mask.sum(), train_mae))
            self.update_node_module.train(False)


            epoch_end_time = time.time()
            epoch_time_duration = epoch_end_time - epoch_start_time

            train_loss = sum([event[0]*event[1]
                              for event in epoch_history["train_loss"]])/sum([event[0] for event in epoch_history["train_loss"]])
            train_mae = sum([event[0]*event[1]
                             for event in epoch_history["train_mae"]])/sum([event[0] for event in epoch_history["train_mae"]])

            history["epoch"].append(epoch)
            history['train_loss'].append(train_loss.cpu())
            history["train_mae"].append(train_mae.cpu())

            if (epoch+1) % report_epoch_steps == 0:
                txt = "epoch: {epoch:.0f}/{epochs:.0f},\n train_loss: {train_loss:.4f},\n train_mae: {train_mae:.4f},\n epoch_time_duration: {epoch_time_duration:.4f}\n"
                print(txt.format(epoch=epoch+1,
                                 epochs=epochs,
                                 train_loss=history['train_loss'][-1],
                                 train_mae=history['train_mae'][-1],
                                 epoch_time_duration=epoch_time_duration),
                      flush=True)

            # change learning rate
            scheduler.step()

            # callbacks
            early_stop = early_stopping_cb.on_epoch_end(
                history, self.update_node_module)
            
            if early_stop:
                print("early stopping activated")
                break

        self.update_node_module.train(False)

        if early_stopping_cb.retrieve_model and early_stopping_cb.model_copy is not None:
            self.update_node_module = early_stopping_cb.model_copy

        print('== end training ==\n')

        return history