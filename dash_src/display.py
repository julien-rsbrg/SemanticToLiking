"""
Keeps all the functions creating figures
"""

import numpy as np
import pandas as pd

import plotly.graph_objects as go

import dash_src.utils as utils

ALL_SAMPLES_COLOR = '#2a8cf5'
COMPLIANT_COLOR = "#5cbd01"
NOT_COMPLIANT_COLOR = "#e80000"

def create_parcoords(df:pd.DataFrame,vars:list[str],color_var:str)->go.Figure:
    """
    Create plotly parallel coordinates graph with color_var as color

    Arguments
    ---------
    - df: (pd.DataFrame)
        Data to plot
    
    - vars: (list[str])
        Variables' names to plot
    
    - color_var: (str)
        Variable name that will define the color of the thread. df[color_var] should be numeric. 
        No need to add color_var in vars; it will be added first automatically.
    
    Returns
    -------
    - go.Figure
        The parallel coordinates figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Parcoords(line = dict(
            color = df[color_var],
            colorscale = 'viridis',
            showscale = True,
            cmin = float(df[color_var].min()),
            cmax = float(df[color_var].max())),
            dimensions = [utils.get_parcoords_dict_dim(df,col) for col in ([color_var]+list(set(vars)-set([color_var])))]))


    fig.update_xaxes(autorange="reversed")
    fig.update_layout(legend=dict(
        orientation="v",
        yanchor="bottom",
        y=1.2,
        xanchor="right",
        x=0.1
    ))
    fig.update_layout(height=500,clickmode='event+select')

    return fig

def create_parcoords_only_histo(df:pd.DataFrame,color_var:str,mask_treatment:np.ndarray)->go.Figure:
    """
    Create the histogram that should be put on the left of a parallel coordinates plot

    (Remark: maybe, adapt height to fit the parcoords alongside)

    Arguments
    ---------
    - df: (pd.DataFrame)
        Data to plot
    
    - color_var: (str)
        Variable name to plot
    
    - mask_treatment: (np.ndarray)
        Flag array. Values that are True are 'constraints compliant'.
    
    Returns
    -------
    - go.Figure
        The vertical histogram figure directed to the left
    """
    fig = go.Figure()
    
    # fig can you get constraintrange: create a if condition if constraintrange in ...
    hist_parcoords = go.Histogram(
        y=df[~mask_treatment][color_var],
        nbinsy=20,
        name="not constraint compliant",
        marker_color=NOT_COMPLIANT_COLOR
        )
    fig.add_trace(hist_parcoords) #secondary_y=True,)


    hist_parcoords = go.Histogram(
        y=df[mask_treatment][color_var],
        nbinsy=20,
        name="constraint compliant",
        marker_color=COMPLIANT_COLOR)
    fig.add_trace(hist_parcoords) #secondary_y=True,)

    hist_parcoords = go.Histogram(
        y=df[color_var],
        nbinsy=20,
        name="all samples",
        marker_color=ALL_SAMPLES_COLOR,
        opacity=0.75
        )
    fig.add_trace(hist_parcoords) #secondary_y=True,)

    fig.update_xaxes(autorange="reversed")
    fig.update_layout(legend=dict(
        orientation="v",
        yanchor="top",
        y=1.3,
        xanchor="left",
        x=0.0
    ))
    fig.update_layout(height=500,clickmode='event+select')
    fig.update_yaxes(title_text=color_var)
    fig.update_xaxes(title_text="count")

    parcoords = go.Figure(data=fig)

    return parcoords


def create_violins(df:pd.DataFrame,vars_name:list[str],mask_treatment:np.ndarray,title:str="")->go.Figure:
    """
    Create a plolty violin plot with vars_name variables

    Arguments
    ---------
    - df: (pd.DataFrame)
        Data to plot
    
    - vars_name: (list[str])
        variables' names to plot
    
    - mask_treatment: (np.ndarray)
        Flag array. Values that are True are 'constraints compliant'.

    Returns
    -------
    - go.Figure
        Violin plot from plotly
    """
    fig = go.Figure()

    for var_id in range(0,len(vars_name)):
        if pd.api.types.is_numeric_dtype(df[vars_name[var_id]]):
            fig.add_trace(go.Violin(#x=[vars_name[var_id]],
                                    y=df[mask_treatment][vars_name[var_id]],
                                    legendgrouptitle_text="Constraints compliant",
                                    legendgroup='constraint compliant', scalegroup='constraint compliant', name=vars_name[var_id],
                                    side='negative',
                                    pointpos=-0.9, # where to position points
                                    line_color='#5cbd01',
                                    showlegend=(var_id==0))
                    )
            fig.add_trace(go.Violin(#x=[vars_name[var_id]],
                                    y=df[~mask_treatment][vars_name[var_id]],
                                    legendgrouptitle_text="Not constraints compliant",
                                    legendgroup='not constraint compliant', scalegroup='not constraint compliant', name=vars_name[var_id],
                                    side='positive',
                                    pointpos=0.9,
                                    line_color='#e80000',
                                    showlegend=(var_id==0))
                    )

    # update characteristics shared by all traces
    fig.update_traces(
        meanline_visible=True,
        points='all', # show all points
        jitter=0.05,  # add some jitter on points for better visibility
        #scalemode='count'
        ) #scale violin plot area with total count
    
    fig.update_layout(
        title_text=title,
        violingap=0, violingroupgap=0, violinmode='overlay')

    return fig


def create_pairplot(df:pd.DataFrame,x_var:str,y_var:str,mask_treatment:np.ndarray|None=None,title:str="")->go.Figure:
    """
    Create a plotly pairplot

    Arguments
    ---------
    - df: (pd.DataFrame)
        Data to plot
    
    - x_var: (str)
        Variable in df to put on x axis
    
    - y_var: (str)
        Variable in df to put on y axis
    
    - mask_treatment: (np.ndarray|None)
        Flag array. Values that are True are 'constraints compliant'. If None, doesn't make a distinction.
    
    - title: (str)
        Title of the plot
    
    Returns
    - go.Figure
        Pairplot plotly
    """
    fig = go.Figure()

    if mask_treatment is None:
        fig.add_trace(go.Scatter(
            x=df[x_var],
            y=df[y_var],
            mode="markers",
            name="All samples",
            marker=dict(color=ALL_SAMPLES_COLOR)
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df[mask_treatment][x_var],
            y=df[mask_treatment][y_var],
            mode="markers",
            name="constraints compliant",
            marker=dict(color=COMPLIANT_COLOR)
        ))
        fig.add_trace(go.Scatter(
            x=df[~mask_treatment][x_var],
            y=df[~mask_treatment][y_var],
            mode="markers",
            name="not constraints compliant",
            marker=dict(color=NOT_COMPLIANT_COLOR)
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_var,
        yaxis_title=y_var,
        clickmode='event+select'
    )
    
    return fig


def create_line_scatter(df:pd.DataFrame,x_var:str,vars_name:list[str],group_name:str,title:str="",fig:go.Figure=None)->go.Figure:
    """
    Create a line scatter or a layer on top of a previous figure (fig)

    Arguments
    ---------
    - df: (pd.DataFrame)
        Data to plot
    
    - x_var: (str)
        variable name on x axis
    
    - vars_name: (list[str])
        Variables to plot
    
    - group_name: (str)
        Group name this layer would belong to in the end figure
        
    - title: (str)
        Title of the plot
    
    - fig: (go.Figure|None)
        If None, creates a new figure. Otherwise, put a layer on top.

    Returns 
    -------
    - go.Figure
        Line scatter plot
    """
    if fig is None:
        fig = go.Figure()
    
    for single_var in vars_name:
        fig.add_trace(go.Scatter(
            x = df[x_var],
            y = df[single_var],
            legendgrouptitle_text=group_name,
            legendgroup=group_name,
            name = single_var,
            mode = "lines"
        ))


    fig.update_layout(
        title=title
    )
    return fig


def create_2d_scatter(
        df:pd.DataFrame,
        x_var:str,
        y_var:str,
        color:any,
        name:str,
        cmin:float|None=None,
        cmax=None,
        color_showscale=False,
        marker_symbol="circle",
        color_bar_title="",
        title="",
        fig=None):
    """
    Create a scatter in 2d or a layer on top of a previous figure (fig)

    Arguments
    ---------
    - df: (pd.DataFrame)
        Data to plot

    - x_var: (str)
        Variable in df to put on x axis
    
    - y_var: (str)
        Variable in df to put on y axis
    
    - color: (np.ndarray|any)
        Color of the markers (can be homogenous if not np.ndarray)
    
    - name: (str)
        Name of this layer
    
    - cmin,cmax: (float)
        Min and max of the color scale
    
    - marker_symbol: (str)
        Symbol of markers
    
    - color_bar_title: (str)
        Title of the color bar
    
    - title: (str)
        Title of the plot
    
    - fig: (go.Figure|None)
        If None, creates a new figure. Otherwise, put a layer on top.

    Returns
    -------
    - go.Figure
        2d scatter plot
    """
    
    if fig is None:
        fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x = df[x_var],
        y = df[y_var],
        name = name,
        mode = "markers",
        marker = dict(
            color=color,
            colorscale='Viridis',
            showscale=color_showscale,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                title=color_bar_title
            ),
            symbol=marker_symbol)
    ))


    fig.update_layout(
        title=title,
        xaxis_title=x_var,
        yaxis_title=y_var
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig


def create_simple_histo(df:pd.DataFrame,var_name:str,nbinsx:int=20,fig:go.Figure|None=None):
    """
    Create a histogram or add it on top of another figure (fig)

    Arguments
    ---------
    - df: (pd.DataFrame)
        Data to plot

    - var_name: (str)
        Variable used for histogram

    - nbinsx: (int)
        Number of bins for histogram
    
    - fig: (go.Figure|None)
        If None, creates a new figure. Otherwise, put a layer on top.

    Returns
    -------
    - go.Figure
        Histogram plot
    """
    if fig is None:
        fig = go.Figure()
    
    hist_parcoords = go.Histogram(
        x=df[var_name],
        nbinsx=nbinsx,
        name=var_name
        )
    fig.add_trace(hist_parcoords)

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=0.0
        )
    )
    fig.update_yaxes(title_text="count")
    fig.update_xaxes(title_text=var_name)

    _fig = go.Figure(data=fig)

    return _fig