"""
Utils functions
"""

import os
import yaml

from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def recursive_mkdirs(folder_path:str)->None:
    """
    Recursively create the folders leading to folder_path

    Arguments
    ---------
    - folder_path: (str)
      path to folder to create as well as its ancestors
    """
    parent_folder_path = os.path.dirname(folder_path)
    if not (os.path.exists(parent_folder_path)) and parent_folder_path != "":
        recursive_mkdirs(parent_folder_path)
        if not(os.path.exists(folder_path)):
            os.makedirs(folder_path)
    else:
        if not(os.path.exists(folder_path)):
            os.makedirs(folder_path)
    
    

def restrict_data(data:pd.DataFrame,group_by,groups_kept,mask_treatment):
  _data = data.copy()
  mask_groups_kept = _data[group_by].isin(groups_kept)
  _data["_mask_treatment"] = mask_treatment
  _data = _data[mask_groups_kept]
  _data.reset_index(inplace=True,drop=True)
  _data = _data.loc[pd.Series(pd.Categorical(_data[group_by],
                                             categories=groups_kept, 
                                             ordered=True)
                                             ).sort_values().index
                                             ]
  mask_treatment = _data["_mask_treatment"].values 
  _data = _data.drop("_mask_treatment",axis=1)
  return _data, mask_treatment




##


def get_parcoords_dict_dim(df:pd.DataFrame,column_name:str,remove_prefix_length:int=0,range:list|None=None)->dict[str]:
    """
    Creates the parameters for a plotly parallel coordinates (parcoords) graph

    Arguments
    ---------
    - df: (pd.DataFrame)
      Data to plot in parcoords
    
    - column_name: (str)
      The specific variable that we need to plot in parcoords

    - remove_prefix_length: (int) 
      Removes the first remove_prefix_length characters in column_name for the parcoords
    
    - range: (list[number]|None)
      The min and max values of df[column_name] we wish to see. If None, it measures it on df.
    
    Returns
    -------
    - dict[str]
      To pass as a list in dimensions of go.Parcoords
    """
    assert len(df)

    if isinstance(df[column_name].iloc[0],str):
        #group_vars = df[column_name].unique()
        dfg = pd.DataFrame({column_name:df[column_name].unique()})
        dfg['dummy'] = dfg.index

        df = pd.merge(df, dfg, on = column_name, how='left')
        return dict(range=[0,df['dummy'].max()],
                   tickvals = dfg['dummy'], ticktext = dfg[column_name],
                   label=column_name[remove_prefix_length:], values=df['dummy'])
    else:
        if range is None:
          print("df[column_name]\n",df[column_name])
          _range = [df[column_name].min(),df[column_name].max()]
        else:
          _range = range
        return dict(range = _range,
                     label = column_name[remove_prefix_length:], values = df[column_name])


def get_mask_treatment_from_parcoords(g_widget:go.Figure,data:pd.DataFrame)->tuple[np.ndarray,set[str]]:
    """
    Read plotly parallel coordinates graph to get the manually applied constraints

    Arguments
    ---------
    - g_widget: (go.Figure)
      The plotly parallel coordinates graph to read
    
    - data: (pd.DataFrame)
      Data that served creating the parcoords graph
    
    Returns
    -------
    - mask_treatment: (np.ndarray)
      Flag array of the rows in data. If True, the row respects the constraints
    
    - treatment_variables: (list[str])
      List of the column names involved in the constraints    
    """
    _data = data.copy()

    mask_treatment = np.ones(_data.shape[0],dtype=bool)
    treatment_variables = set()
    for dimension in g_widget["data"][0]["dimensions"]:
      feature_name = dimension["label"]
      if pd.api.types.is_string_dtype(_data[feature_name]):
        _feature_name = feature_name+"_dummy"
        _data.insert(len(_data.columns),_feature_name,get_parcoords_dict_dim(_data,feature_name)["values"].values,False)
        _data = _data.reset_index(drop=True) # adding the new column killed the index...
      else:
        _feature_name = feature_name

      mask_feature = np.zeros(_data.shape[0],dtype=bool)
      if ("constraintrange" in dimension) and (dimension["constraintrange"] is not None) and len(dimension["constraintrange"]):
        if not(isinstance(dimension["constraintrange"][0],Iterable)):
          (bound_lw,bound_up) =  dimension["constraintrange"]
          mask_feature+= (_data[_feature_name]>=bound_lw)*(_data[_feature_name]<=bound_up)
        else:
          for (bound_lw,bound_up) in dimension["constraintrange"]:
            mask_feature+= (_data[_feature_name]>=bound_lw)*(_data[_feature_name]<=bound_up)
        mask_treatment*=mask_feature
        treatment_variables |= set([feature_name])
    return mask_treatment,treatment_variables


def extract_range(data:pd.DataFrame,var_name:str,min_bound:float|int,max_bound:float|int)->pd.DataFrame:
  """
  Take the rows in data with variable var_name in between min_bound and max_bound

  Arguments
  ---------
  - data: (pd.DataFrame)
    Original data to extract from
  
  - var_name: (str)
    Variable along which the restriction will be applied
  
  - min_bound,max_bound: (number)
    Bounds between which the variable should be in. 

  Returns
  -------
  - (pd.DataFrame)
    Extracted dataframe
  
  """

  mask = (data[var_name]>=min_bound)*(data[var_name]<=max_bound)
  return data[mask]


def get_slider_params(var_values:pd.Series,ticks_per_range:int,n_potential_values:int=None)->tuple[float|dict]:
  """
  Infer the slider parameters of dcc.Slider from the values

  Arguments
  ---------
  - var_values: (pd.Series)
    Variable values
  
  - ticks_per_range: (int)
    Number of ticks to put on the marks
  
  - n_potential_values: (int|None)
    Number of values allowed for the slider. Would restrict the step parameter.

  Returns
  -------
  - vmin,vmax: (float)
    minimum and maximum values in var_values
  
  - step: (float)
    minimal step allowed in the slider
  
  - marks: (dict[str,float])
    Marks where ticks are written for helping the user 
  """
  vmin,vmax = var_values.min(),var_values.max()
  tick_vals = np.linspace(vmin,vmax,num=ticks_per_range,endpoint=True)

  is_int_ticks = False
  if (n_potential_values is None) or (n_potential_values <= 1):
    step = var_values.diff().min()
    is_int_ticks = (var_values.dtype in [np.int64,np.int32])
  else:
    step = (vmax-vmin)/(n_potential_values-1)

  if is_int_ticks:
    marks = {int(v):"%d"%int(v) for v in tick_vals}
  else:
    marks = {v:"%.2f"%v for v in tick_vals}

  return vmin,vmax,step,marks


