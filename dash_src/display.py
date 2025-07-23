## Keeps all the functions creating figures



import numpy as np
import pandas as pd
from scipy import stats


import plotly.graph_objects as go

import dash_src.utils as utils

ALL_SAMPLES_COLOR = '#2a8cf5'
ALL_SAMPLES_SYMBOL = "circle"
COMPLIANT_COLOR = "#5cbd01"
COMPLIANT_SYMBOL = "x"
NOT_COMPLIANT_COLOR = "#e80000"
NOT_COMPLIANT_SYMBOL = "triangle-down"

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


def create_pairplot(
        df:pd.DataFrame,
        x_var:str,
        y_var:str,
        mask_treatment:np.ndarray|None=None,
        group_by:str|None = None,
        title:str="",
        show_legend:bool=True)->go.Figure:
    """
    Create a plotly pairplot

    TODO: merge with plot_2d_scatter

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

    - group_by: (...)
        TODO
    
    - title: (str)
        Title of the plot
    
    Returns
    -------
    - go.Figure
        Pairplot plotly
    """
    assert group_by is None or group_by in df

    fig = go.Figure()

    if mask_treatment is None:
        if group_by is None:
            fig.add_trace(go.Scatter(
                x=df[x_var],
                y=df[y_var],
                mode="markers",
                name="all samples",
                marker=dict(color=ALL_SAMPLES_COLOR),
                showlegend=True
            ))
        else:
            for i,v in enumerate(df[group_by].unique()):
               mask = df[group_by] == v
               fig.add_trace(go.Scatter(
                    x=df[mask][x_var],
                    y=df[mask][y_var],
                    mode="markers",
                    name=f"all samples -- {group_by}: {v}",
                    marker=dict(symbol=ALL_SAMPLES_SYMBOL),
                    showlegend=True
                )) 
    else:
        if group_by is None:
            fig.add_trace(go.Scatter(
                x=df[mask_treatment][x_var],
                y=df[mask_treatment][y_var],
                mode="markers",
                name="constraints compliant",
                marker=dict(color=COMPLIANT_COLOR),
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=df[~mask_treatment][x_var],
                y=df[~mask_treatment][y_var],
                mode="markers",
                name="not constraints compliant",
                marker=dict(color=NOT_COMPLIANT_COLOR)
            ))
        else:
            for i,v in enumerate(df[group_by].unique()):
                mask = df[group_by] == v
                fig.add_trace(go.Scatter(
                    x=df[mask_treatment * mask][x_var],
                    y=df[mask_treatment * mask][y_var],
                    mode="markers",
                    name=f"constraints compliant -- {group_by}: {v}",
                    marker=dict(symbol=COMPLIANT_SYMBOL),
                    showlegend=True
                ))
                fig.add_trace(go.Scatter(
                    x=df[(~mask_treatment) * mask][x_var],
                    y=df[(~mask_treatment) * mask][y_var],
                    mode="markers",
                    name=f"not constraints compliant -- {group_by}: {v}",
                    marker=dict(symbol=NOT_COMPLIANT_SYMBOL)
                ))

    fig.update_layout(
        title=title,
        xaxis_title=x_var,
        yaxis_title=y_var,
        clickmode='event+select',
        showlegend=show_legend
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


def create_violin_distrib(
        df: pd.DataFrame,
        var_x: str, 
        var_y: str, 
        mask_treatment: np.ndarray|None = None, 
        group_by: str|None = None,
        title: str = "",
        show_legend: bool = True):
    fig = go.Figure()

    if mask_treatment is None:
        if group_by is None:
            fig.add_trace(
                go.Violin(
                    x=df[var_x],
                    y=df[var_y],
                    pointpos=-1,
                    showlegend=True
                )
            )
        else:
            for v in df[group_by].unique():
                mask = df[group_by] == v
                fig.add_trace(
                    go.Violin(
                        x=df[mask][var_x],
                        y=df[mask][var_y],
                        legendgroup=v, 
                        scalegroup=v, 
                        name=v,
                        pointpos=-1,
                        showlegend=True
                    )
                )
            
    else:
        if group_by is None:
            fig.add_trace(
                go.Violin(
                    x=df[mask_treatment][var_x],
                    y=df[mask_treatment][var_y],
                    legendgroup="constraints compliant", 
                    scalegroup="constraints compliant", 
                    name="constraints compliant",
                    line_color=COMPLIANT_COLOR,
                    pointpos=-1,
                    showlegend=True
                )
            )
            fig.add_trace(
                go.Violin(
                    x=df[~mask_treatment][var_x],
                    y=df[~mask_treatment][var_y],
                    legendgroup="not constraints compliant", 
                    scalegroup="not constraints compliant", 
                    name="not constraints compliant",
                    line_color=NOT_COMPLIANT_COLOR,
                    pointpos=-1
                )
            )
        else:
            for v in df[group_by].unique():
                mask = df[group_by] == v
                fig.add_trace(
                    go.Violin(
                        x=df[mask_treatment * mask][var_x],
                        y=df[mask_treatment * mask][var_y],
                        legendgroup=f"constraints compliant -- {group_by}: {v}", 
                        scalegroup=f"constraints compliant -- {group_by}: {v}", 
                        name=f"constraints compliant -- {group_by}: {v}",
                        #line_color=COMPLIANT_COLOR,
                        pointpos=-1,
                        showlegend=True
                    )
                )
                fig.add_trace(
                    go.Violin(
                        x=df[(~mask_treatment) * mask][var_x],
                        y=df[(~mask_treatment) * mask][var_y],
                        legendgroup=f"not constraints compliant -- {group_by}: {v}", 
                        scalegroup=f"not constraints compliant -- {group_by}: {v}", 
                        name=f"not constraints compliant -- {group_by}: {v}",
                        #line_color=NOT_COMPLIANT_COLOR,
                        pointpos=-1
                    )
                )

    fig.update_traces(
        box_visible=True, 
        meanline_visible=True,
        jitter=0.1,
        points="all")
    fig.update_layout(
        title = title,
        violinmode='group',
        xaxis_title=var_x,
        yaxis_title=var_y,
        showlegend=show_legend)
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




def create_grouped_scatter_plot_matrix(data:pd.DataFrame,var:str,join_on:str,group_by:str,join_on_aggr:str="mean",title:str=""):
    joined_data = utils.refactor_data(data = data, var = var, join_on = join_on, group_by = group_by, join_on_aggr = join_on_aggr)

    # collect data
    dimensions = []
    for v_group in data[group_by].unique():
        dimensions.append({"label":v_group,"values":joined_data[var+":"+v_group]})

    # flesh out the figure
    fig = go.Figure(data=go.Splom(
        dimensions=dimensions,
        diagonal_visible=False, # remove plots on diagonal
        text=join_on + ": " + joined_data[join_on].astype(str),
        marker=dict(line_color='white', line_width=0.5)))

    fig.update_layout(title_text=title)

    return fig



def create_grouped_heatmap(data:pd.DataFrame,var:str,fn:str,join_on:str,group_by:str,join_on_aggr:str="mean",title:str="",showticks:bool=True):
    """
    A

    Arguments
    ---------
    - join_on_aggr : (str), default 'mean'
        choose how to aggregate the values of var for when there are several values for a group and a value of join_on
        
    """
    possible_group_values = data[group_by].unique()

    joined_data = utils.refactor_data(data = data, var = var, join_on = join_on, group_by = group_by, join_on_aggr = join_on_aggr)

    # apply function
    z = []
    p_values = [] # when used
    if fn == "mean difference y-x":
        print("mean diff")
        for v_group_y in possible_group_values:
            new_row = []
            for v_group_x in possible_group_values:
                print("v_group_x, v_group_y", v_group_x, v_group_y)
                diff = np.mean(joined_data[var+":"+str(v_group_y)] - joined_data[var+":"+str(v_group_x)])
                new_row.append(diff)
            z.append(new_row)

    elif fn == "mean absolute difference y-x":
        print("mean diff")
        for v_group_y in possible_group_values:
            new_row = []
            for v_group_x in possible_group_values:
                print("v_group_x, v_group_y", v_group_x, v_group_y)
                diff = np.abs(np.mean(joined_data[var+":"+str(v_group_y)] - joined_data[var+":"+str(v_group_x)]))
                new_row.append(diff)
            z.append(new_row)

    elif fn == "correlation (Pearson)":
        for v_group_y in possible_group_values:
            new_row = []
            new_row_p = []
            for v_group_x in possible_group_values:
                mask_na = ~(joined_data[var+":"+str(v_group_y)].isna() + joined_data[var+":"+str(v_group_x)].isna())
                if len(joined_data[mask_na]) >=2:
                    res = stats.pearsonr(joined_data[mask_na][var+":"+str(v_group_y)], joined_data[mask_na][var+":"+str(v_group_x)])
                    new_row.append(res.statistic)
                    new_row_p.append(res.pvalue)
                else:
                    new_row.append(np.nan)
                    new_row_p.append(np.nan)
            z.append(new_row)
            p_values.append(new_row_p)

    elif fn == "correlation (Spearman)":
        for v_group_y in possible_group_values:
            new_row = []
            new_row_p = []
            for v_group_x in possible_group_values:
                mask_na = ~(joined_data[var+":"+str(v_group_y)].isna() + joined_data[var+":"+str(v_group_x)].isna())
                if len(joined_data[mask_na]) >=2:
                    res = stats.spearmanr(joined_data[mask_na][var+":"+str(v_group_y)], joined_data[mask_na][var+":"+str(v_group_x)])
                    new_row.append(res.statistic)
                    new_row_p.append(res.pvalue)
                else:
                    new_row.append(np.nan)
                    new_row_p.append(np.nan)
            z.append(new_row)
            p_values.append(new_row_p)
    
    elif fn == "coefficient of determination y w.r.t. x":
        pass # need true values
    else:
        raise NotImplementedError(f"Function {fn} is unknown")


    # flesh out the figure
    if len(p_values):
        # p_values in use
        text = pd.DataFrame(z).map('{:.3f}'.format).astype(str) + " - p:"+ pd.DataFrame(p_values).map('{:.3f}'.format).astype(str)
    else:
        text = pd.DataFrame(z).map('{:.3f}'.format).astype(str)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=data[group_by].unique(),
        y=data[group_by].unique(),
        text=text,
        texttemplate="%{text}"))

    fig.update_layout(title_text=title,
                      xaxis_showticklabels=showticks,
                      yaxis_showticklabels=showticks)

    return fig



def plot_pairplot_matrix(
        data: pd.DataFrame, 
        var_names: list[str], 
        color_var_name:str|None = None, 
        text_var_name:str|None = None, 
        title = ""):
    
    dimensions = [{"label":name, "values":data[name]} for name in var_names]

    text = None
    if not(text_var_name is None):
        coerced_numeric_df = pd.to_numeric(data[text_var_name], errors='coerce') 
        if ((~coerced_numeric_df.isna()) + data[text_var_name].isna()).sum() == len(data[text_var_name]):
            # every value is nan or numeric
            text = text_var_name + ": " + data[text_var_name].map('{:.3f}'.format).astype(str)
        else:
            text = text_var_name +": " + data[text_var_name].astype(str)

    fig = go.Figure(data=go.Splom(
                    dimensions=dimensions,
                    diagonal_visible=False, # remove plots on diagonal
                    text = text,
                    marker=dict(color=data[color_var_name] if not(color_var_name is None) else None,
                                showscale=True,
                                colorscale='Viridis', 
                                line_color='white',
                                line_width=0.5,
                                colorbar = {"title":color_var_name}),
                    
                    ))

    fig.update_layout(
        title_text = title
    )

    return fig