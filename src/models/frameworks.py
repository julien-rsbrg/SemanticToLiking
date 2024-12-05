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

import src.models.utils.callbacks as callbacks
import src.models.utils.common as common_utils


class GNN_naive_framework:
    # Class initialization params
    def __init__(self, 
                 update_node_module, 
                 device, 
                 mask_node_fn:Optional[Callable] = None, 
                 mask_node_attr:Optional[torch.Tensor] = None):
        self.device = device
        self.update_node_module = update_node_module.to(self.device)
        
        self.mask_node_fn = mask_node_fn
        self.mask_node_attr = mask_node_attr


    # Compute the solution of the PDE on the points (x,y)
    def predict(self, node_attr, edge_index, edge_attr, **kwargs):
        node_attr = node_attr.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        if not((self.mask_node_fn is None) and (self.mask_node_attr is None)):
            _node_attr = common_utils.replace_by_value(
                node_attr,
                mask_samples=self.mask_node_fn(node_attr),
                mask_attr=self.mask_node_attr
            )
        else:
            _node_attr = node_attr.clone()

        model_out = self.update_node_module(_node_attr, edge_index, edge_attr, **kwargs)
        return model_out

    # Compute the loss function   
    def loss_function(self, node_attr, edge_index, edge_attr, labels, mask):
        node_attr = node_attr.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        preds = self.predict(node_attr, edge_index, edge_attr)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(preds[mask], labels[mask])
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

    def train_step(self, node_attr, edge_index, edge_attr, labels, train_mask, optimizer):
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Compute the loss and its gradients
        # mask applied after so that neighbors are taken into account
        loss = self.loss_function(node_attr, edge_index, edge_attr, labels, train_mask)
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

    def train(self, train_loader, epochs, report_epoch_steps, optimizer, scheduler, early_stopping_monitor="val_mae", patience=torch.inf, min_delta=1e-5, verbose=False):
        """Training loop."""

        print("== start training ==")
        # Initialize the optimizer.
        history = {"train_loss": [], 
                   "train_mae": [],
                   "val_loss": [], 
                   "val_mae": []}
        early_stopping_cb = callbacks.EarlyStopping(
            early_stopping_monitor, patience, min_delta=min_delta)

        for epoch in range(epochs):
            epoch_history = {k: [] for k in history.keys()}
            epoch_start_time = time.time()

            # training
            self.update_node_module.train(True)
            for batch_i, batch_graph in enumerate(train_loader):
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
                                     n_batch=len(train_loader),
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