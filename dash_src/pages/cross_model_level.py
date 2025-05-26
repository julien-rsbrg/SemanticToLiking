import copy 
import os

import numpy as np
import pandas as pd

from dash import dcc, html, Input, Output, State, callback, register_page, dash_table
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

import dash_src.display as display
import dash_src.utils as utils
import dash_src.data_load as data_load

from dash_src.configs.main import config

from src.utils import read_yaml

register_page(__name__, path="/")

## Data

CONFIG = {
    "src_results_path":"src/data_generation/examples",
}
RAW_PATH = os.path.join(CONFIG["src_results_path"],"raw")
PROCESSED_PATH = os.path.join(CONFIG["src_results_path"],"processed")

id_to_models_info, all_studies_summaries = data_load.load_data(processed_path=PROCESSED_PATH)



################################################################


## Layout

layout = html.Div(
    [   
        dcc.Store("cross_model_store_mask_treatment",data=[]),
        
        html.H2("Cross Model Level"),

        ### Models selection
        html.H3("Model selection"),
        html.Label("Retrieve information on the models"),
        dcc.Dropdown(
            options=[id_to_models_info[i]["model_id"] for i in id_to_models_info.keys()],
            value=id_to_models_info[0]["model_id"], # at least 1 study stored
            id='cross_model_info_selected',
            multi=False
        ),
        html.H5("Model information:"),
        dcc.Markdown("...model info...",id="cross_model_info_report",style={"width":"80%"}),

        html.H4("Choose the models for comparison"),
        html.Label("(beware of the order: it will be used for the plots)"),
        dcc.Dropdown(
            options=[id_to_models_info[i]["model_id"] for i in id_to_models_info.keys()],
            value=[id_to_models_info[0]["model_id"]],
            id='models_comparison_selected',
            multi=True
        ),

        
        

        ### Data Selection
        html.H3("Data selection"),

        html.H4("Parcoords"),
        html.Label("Choose the variables"),
        dcc.Dropdown(
            options=all_studies_summaries.columns.tolist(),
            value=all_studies_summaries.columns.tolist()[:2],
            id='cross_model_parcoords_vars',
            multi=True
        ),

        html.Label("Choose the color variable"),
        dcc.Dropdown(
            options=all_studies_summaries.select_dtypes(include=np.number).columns.tolist(),
            value=all_studies_summaries.select_dtypes(include=np.number).columns.tolist()[0],
            id='cross_model_parcoords_color'
        ),
        html.Div([
            html.Div([
                dcc.Graph(id="cross_model_histo_parcoords")
                ],
                style={'width': '15%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id="cross_model_parcoords")
                ],
                style={'width': '85%', 'display': 'inline-block'}),
        ]),

        html.H3("Comparison"),

        html.H4("Pair plots"),

        html.Div(
            [
            html.Div([
                html.Div([
                    html.Label("Choose var y"),
                    dcc.Dropdown(
                        options = all_studies_summaries.select_dtypes(include=np.number).columns.tolist(),
                        value = all_studies_summaries.select_dtypes(include=np.number).columns.tolist()[0],
                        id = "cross_model_scatter_0_var_y"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Choose var x"),
                    dcc.Dropdown(
                        options = all_studies_summaries.select_dtypes(include=np.number).columns.tolist(),
                        value = all_studies_summaries.select_dtypes(include=np.number).columns.tolist()[0],
                        id = "cross_model_scatter_0_var_x"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                dcc.RadioItems(['mix', 'use constraints'], 'mix', id="cross_model_scatter_0_constraints",inline=True),
                dcc.RadioItems(['mix', 'group by model'], 'mix', id="cross_model_scatter_0_model_group",inline=True),
                dcc.Graph(id="cross_model_scatter_0"),
            ],style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.Div([
                    html.Label("Choose var y"),
                    dcc.Dropdown(
                        options = all_studies_summaries.select_dtypes(include=np.number).columns.tolist(),
                        value = all_studies_summaries.select_dtypes(include=np.number).columns.tolist()[0],
                        id = "cross_model_scatter_1_var_y"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Choose var x"),
                    dcc.Dropdown(
                        options = all_studies_summaries.select_dtypes(include=np.number).columns.tolist(),
                        value = all_studies_summaries.select_dtypes(include=np.number).columns.tolist()[0],
                        id = "cross_model_scatter_1_var_x"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                dcc.RadioItems(['mix', 'use constraints'], 'mix', id="cross_model_scatter_1_constraints",inline=True),
                dcc.RadioItems(['mix', 'group by model'], 'mix', id="cross_model_scatter_1_model_group",inline=True),
                dcc.Graph(id="cross_model_scatter_1"),
            ],style={'width': '45%', 'display': 'inline-block'}),
            ]
        ),

        ### Core comparison

        html.H4("Distributions"),

        html.Div([
            html.Div([
                html.Div([
                    html.Label("Choose var y"),
                    dcc.Dropdown(
                        options = all_studies_summaries.select_dtypes(include=np.number).columns.tolist(),
                        value = all_studies_summaries.select_dtypes(include=np.number).columns.tolist()[0],
                        id = "cross_model_distrib_0_var_y"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Choose var x"),
                    dcc.Dropdown(
                        options = all_studies_summaries.select_dtypes(include=np.number).columns.tolist(),
                        value = all_studies_summaries.select_dtypes(include=np.number).columns.tolist()[0],
                        id = "cross_model_distrib_0_var_x"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                dcc.RadioItems(['mix', 'use constraints'], 'mix', id="cross_model_distrib_0_constraints",inline=True),
                dcc.RadioItems(['bar plot', 'violin plot'], 'bar plot', id="cross_model_distrib_0_bar_or_violin",inline=True),
                dcc.Graph(id="cross_model_distrib_0")
            ], style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.Div([
                    html.Label("Choose var y"),
                    dcc.Dropdown(
                        options = all_studies_summaries.select_dtypes(include=np.number).columns.tolist(),
                        value = all_studies_summaries.select_dtypes(include=np.number).columns.tolist()[0],
                        id = "cross_model_distrib_1_var_y"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Choose var x"),
                    dcc.Dropdown(
                        options = all_studies_summaries.select_dtypes(include=np.number).columns.tolist(),
                        value = all_studies_summaries.select_dtypes(include=np.number).columns.tolist()[0],
                        id = "cross_model_distrib_1_var_x"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                dcc.RadioItems(['mix', 'use constraints'], 'mix', id="cross_model_distrib_1_constraints",inline=True),
                dcc.RadioItems(['bar plot', 'violin plot'], 'bar plot', id="cross_model_distrib_1_bar_or_violin",inline=True),
                dcc.Graph(id="cross_model_distrib_1")
            ], style={'width': '45%', 'display': 'inline-block'}),
        ]),

        html.H4("Matrices"),

        html.Div([
            html.Div([
                html.Div([
                    html.H5("Real data"),
                    html.Label("Choose variable"),
                    dcc.Dropdown(
                        options = all_studies_summaries.select_dtypes(include=np.number).columns.tolist(),
                        value = all_studies_summaries.select_dtypes(include=np.number).columns.tolist()[0],
                        id = "cross_model_matrix_0_var"
                    ),
                ], style={'width': '80%', 'display': 'inline-block'}),
                dcc.RadioItems(['scatter', 'heatmap'], "scatter", id="cross_model_matrix_0_scatter_vs_heatmap",inline=True),
                html.Div([
                    html.Label("Choose formula for heatmap"),
                    dcc.Dropdown(
                        options = ["mean difference y-x", "mean absolute difference y-x", "correlation (Pearson)", "correlation (Spearman)"],
                        value = "mean difference y-x",
                        id = "cross_model_matrix_0_heatmap_formula"
                    ),
                ], style={'width': '80%', 'display': 'inline-block'}),
                dcc.Graph(id="cross_model_matrix_0")
            ], style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.Div([
                    html.H5("Generated data (by model in x-axis)"),
                    html.Label("Choose variable"),
                    dcc.Dropdown(
                        options = all_studies_summaries.select_dtypes(include=np.number).columns.tolist(),
                        value = all_studies_summaries.select_dtypes(include=np.number).columns.tolist()[0],
                        id = "cross_model_matrix_1_var"
                    ),
                ], style={'width': '80%', 'display': 'inline-block'}),
                dcc.RadioItems(['scatter', 'heatmap'], "scatter", id="cross_model_matrix_1_scatter_vs_heatmap",inline=True),
                dcc.Graph(id="cross_model_matrix_1")
            ], style={'width': '45%', 'display': 'inline-block'}),
        ]),


    ]
)


## Callbacks   
@callback(
    Output("cross_model_info_report","children"),
    Input("cross_model_info_selected","value")
)
def give_model_info(model_name):
    for i in id_to_models_info.keys():
        if model_name == id_to_models_info[i]["model_id"]:
            text = ""
            for k,v in id_to_models_info[i]["constant_config"].items():
                text += str(k) + ": "
                if isinstance(v,list):
                    text += "\n"
                    for v_elem in v:
                        text += "- "+str(v_elem)+"\n"
                else:
                    text += str(v)
                text += "\n\n"

            return text
    return ""


# parcoords

@callback(
    Output("cross_model_store_mask_treatment","data"),
    Input("cross_model_parcoords","restyleData"),
    Input("cross_model_parcoords","figure")
)
def update_cross_model_mask_treatment(restyle_data,fig_parcoords,data=all_studies_summaries):
    mask_treatment, _ = utils.get_mask_treatment_from_parcoords(fig_parcoords,data)
    return mask_treatment

@callback(
    Output("cross_model_histo_parcoords","figure"),
    Input("cross_model_parcoords_color","value"),
    Input("cross_model_store_mask_treatment","data"),
    Input("models_comparison_selected","value")
)
def update_cross_model_only_histo(color_var_name,mask_treatment,groups_kept,group_by="model_name",data=all_studies_summaries):
    data, mask_treatment = utils.restrict_data(data,group_by,groups_kept,np.array(mask_treatment))
    return display.create_parcoords_only_histo(data,color_var_name,mask_treatment)


@callback(
    Output("cross_model_parcoords","figure"),
    Input("cross_model_parcoords_vars","value"),
    Input("cross_model_parcoords_color","value"),
    Input("models_comparison_selected","value")
)
def update_cross_model_parcoords(vars_name,color_var_name,groups_kept,group_by="model_name",data=all_studies_summaries):
    data, _ = utils.restrict_data(data,group_by,groups_kept,np.ones(len(data),dtype=bool))
    return display.create_parcoords(data,vars=vars_name,color_var=color_var_name)



# pair plots

@callback(
    Output("cross_model_scatter_0","figure"),
    Input("cross_model_scatter_0_var_x","value"),
    Input("cross_model_scatter_0_var_y","value"),
    Input("cross_model_scatter_0_constraints","value"),
    Input("cross_model_store_mask_treatment","data"),
    Input("cross_model_scatter_0_model_group","value"),
    Input("models_comparison_selected","value")
)
def update_cross_model_scatter_0(xaxis_var_name,yaxis_var_name,constraints,mask_treatment,group,groups_kept,group_by="model_id",data=all_studies_summaries):
    data, mask_treatment = utils.restrict_data(data,group_by,groups_kept,np.array(mask_treatment))
    return display.create_pairplot(
        data,
        xaxis_var_name,
        yaxis_var_name,
        mask_treatment = None if constraints != "use constraints" else mask_treatment,
        group_by = None if group != "group by model" else group_by)


@callback(
    Output("cross_model_scatter_1","figure"),
    Input("cross_model_scatter_1_var_x","value"),
    Input("cross_model_scatter_1_var_y","value"),
    Input("cross_model_scatter_1_constraints","value"),
    Input("cross_model_store_mask_treatment","data"),
    Input("cross_model_scatter_1_model_group","value"),
    Input("models_comparison_selected","value")
)
def update_cross_model_scatter_1(xaxis_var_name,yaxis_var_name,constraints,mask_treatment,group,groups_kept,group_by="model_id",data=all_studies_summaries):
    data, mask_treatment = utils.restrict_data(data,group_by,groups_kept,np.array(mask_treatment))
    return display.create_pairplot(
        data,
        xaxis_var_name,
        yaxis_var_name,
        mask_treatment = None if constraints != "use constraints" else mask_treatment,
        group_by = None if group != "group by model" else group_by)


# distributions

@callback(
    Output("cross_model_distrib_0","figure"),
    Input("cross_model_distrib_0_var_x","value"),
    Input("cross_model_distrib_0_var_y","value"),
    Input("cross_model_distrib_0_constraints","value"),
    Input("cross_model_distrib_0_bar_or_violin","value"),
    Input("cross_model_store_mask_treatment","data"),
    Input("models_comparison_selected","value")
)
def update_cross_model_distrib_0(var_x,var_y,constraints,bar_or_violin,mask_treatment,groups_kept,group_by="model_id",data=all_studies_summaries):    
    data, mask_treatment = utils.restrict_data(data,group_by,groups_kept,np.array(mask_treatment))
    if bar_or_violin == "violin plot":
        return display.create_violin_distrib(
            df=data,
            var_x=var_x, 
            var_y=var_y, 
            mask_treatment = None if constraints != "use constraints" else mask_treatment, 
            group_by = group_by,
            title = "")
    else:
        return go.Figure()


@callback(
    Output("cross_model_distrib_1","figure"),
    Input("cross_model_distrib_1_var_x","value"),
    Input("cross_model_distrib_1_var_y","value"),
    Input("cross_model_distrib_1_constraints","value"),
    Input("cross_model_distrib_1_bar_or_violin","value"),
    Input("cross_model_store_mask_treatment","data"),
    Input("models_comparison_selected","value")
)
def update_cross_model_distrib_1(var_x,var_y,constraints,bar_or_violin,mask_treatment,groups_kept,group_by="model_id",data=all_studies_summaries):    
    data, mask_treatment = utils.restrict_data(data,group_by,groups_kept,np.array(mask_treatment))
    if bar_or_violin == "violin plot":
        return display.create_violin_distrib(
            df=data,
            var_x=var_x, 
            var_y=var_y, 
            mask_treatment = None if constraints != "use constraints" else mask_treatment, 
            group_by = group_by,
            title = "")
    else:
        return go.Figure()

# matrices
@callback(
    Output("cross_model_matrix_0","figure"),
    Input("cross_model_matrix_0_var","value"),
    Input("cross_model_matrix_0_scatter_vs_heatmap","value"),
    Input("cross_model_matrix_0_heatmap_formula","value"),
    Input("models_comparison_selected","value")
)
def update_cross_model_matrix_0(var,scatter_vs_heatmap,heatmap_formula,groups_kept,join_on="participant_folder_name",group_by="model_id",data=all_studies_summaries):
    data, _ = utils.restrict_data(data,group_by,groups_kept,np.zeros(len(data),dtype=bool))
    
    if scatter_vs_heatmap == "scatter":
        return display.create_grouped_scatter_plot_matrix(
            data=data,
            var=var,
            join_on=join_on,
            group_by=group_by
        )
    else:
        return display.create_grouped_heatmap(
            data=data,
            var=var,
            fn=heatmap_formula,
            join_on=join_on,
            group_by=group_by
        )