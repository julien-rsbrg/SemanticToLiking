import copy 

import numpy as np
import pandas as pd

from dash import dcc, html, Input, Output, State, callback, register_page, dash_table
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

import dash_src.display as display
import dash_src.utils as utils
import dash_src.data_load as data_load

from dash_src.configs.main import config

register_page(__name__, path="/")

## Data

cross_exp_aggregated_results = pd.DataFrame({"a":[1,2,3],"b":[5,3,2]})
participants_data = pd.DataFrame({"ID":[1,2],"depression_score":[1,5],"n_experienced":[4,5],"n_not_experience":[10,9]})
model_to_participant_results = {
    "bayesian_NN":pd.DataFrame({"ID":[1,2],"MAE":[0.2,0.15]}),
    "3NN":pd.DataFrame({"ID":[1,2],"MAE":[0.3,0.2]}),
    "bayesian_GNN":pd.DataFrame({"ID":[1,2],"MAE":[0.21,0.19]})
}

## Layout

layout = html.Div(
    [   
        dcc.Store("cross_model_store_mask_treatment",data=[]),
        dcc.Store("cross_model_store_selected_indices",data=[]),

        
        html.H2("Cross Model Level"),

        ### Models selection
        html.H3("Model selection"),
        html.Label("Retrieve information on the models"),
        dcc.Dropdown(
            options=list(model_to_participant_results.keys()),
            value=list(model_to_participant_results.keys())[0],
            id='cross_model_info_selected',
            multi=False
        ),
        html.H5("Model information:"),
        dcc.Markdown("blabla",id="model_info_report",style={"width":"80%"}),

        html.H4("Choose the models for comparison"),
        dcc.Dropdown(
            options=list(model_to_participant_results.keys()),
            value=list(model_to_participant_results.keys())[:2],
            id='models_comparison_selected',
            multi=True
        ),

        
        

        ### Data Selection
        html.H3("Data selection"),

        html.H4("Parcoords"),
        html.Label("Choose the variables"),
        dcc.Dropdown(
            options=participants_data.columns.tolist(),
            value=participants_data.columns.tolist()[:2],
            id='cross_model_parcoords_dropdown',
            multi=True
        ),

        html.Label("Choose the color variable"),
        dcc.Dropdown(
            options=participants_data.select_dtypes(include=np.number).columns.tolist(),
            value=participants_data.select_dtypes(include=np.number).columns.tolist()[0],
            id='cross_model_color_parcoords_dropdown'
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

        html.H4("Distributions depending on participants' features"),
        html.Div(
            [
            html.Div([
                html.Div([
                    html.Label("Choose var x"),
                    dcc.Dropdown(
                        options = ["var0","var1"],
                        value = "var0",
                        id = "cross_model_scatter_0_var_x"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Choose var y"),
                    dcc.Dropdown(
                        options = ["var0","var1"],
                        value = "var0",
                        id = "cross_model_scatter_0_var_y"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                dcc.RadioItems(['all', 'separate'], 'separate', id="cross_model_scatter_0_constraint",inline=True),
                dcc.Graph(id="cross_model_scatter_0"),
            ],style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.Div([
                    html.Label("Choose var x"),
                    dcc.Dropdown(
                        options = ["var0","var1"],
                        value = "var0",
                        id = "cross_model_scatter_1_var_x"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Choose var y"),
                    dcc.Dropdown(
                        options = ["var0","var1"],
                        value = "var0",
                        id = "cross_model_scatter_1_var_y"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                dcc.RadioItems(['all', 'separate'], 'separate', id="cross_model_scatter_1_constraint",inline=True),
                dcc.Graph(id="cross_model_scatter_1"),
            ],style={'width': '45%', 'display': 'inline-block'}),
            ]
        ),

        ### Core comparison

        html.H3("Comparison"),

        html.Div([
            html.Div([
                html.Div([
                    html.Label("Choose var y"),
                    dcc.Dropdown(
                        options = ["var0","var1"],
                        value = "var0",
                        id = "cross_model_distrib_0_var_y"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Choose var x"),
                    dcc.Dropdown(
                        options = ["var0","var1"],
                        value = "var0",
                        id = "cross_model_distrib_0_var_x"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                dcc.RadioItems(['all', 'only constraint compliant', 'only constraint not compliant'], 'all', id="cross_model_distrib_0_constraint",inline=True),
                dcc.RadioItems(['bar plot', 'violin plot'], 'bar plot', id="cross_model_distrib_0_bar_vs_violin",inline=True),
                dcc.Graph(id="cross_model_distrib_0")
            ], style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.Div([
                    html.Label("Choose var y"),
                    dcc.Dropdown(
                        options = ["var0","var1"],
                        value = "var0",
                        id = "cross_model_distrib_1_var_y"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Choose var x"),
                    dcc.Dropdown(
                        options = ["var0","var1"],
                        value = "var0",
                        id = "cross_model_distrib_1_var_x"
                    ),
                ], style={'width': '45%', 'display': 'inline-block'}),
                dcc.RadioItems(['all', 'only constraint compliant', 'only constraint not compliant'], 'all', id="cross_model_distrib_1_constraint",inline=True),
                dcc.RadioItems(['bar plot', 'violin plot'], 'bar plot', id="cross_model_distrib_1_bar_vs_violin",inline=True),
                dcc.Graph(id="cross_model_distrib_1")
            ], style={'width': '45%', 'display': 'inline-block'}),
        ]),


        html.Div([
            html.Div([
                html.Div([
                    html.H5("Real data"),
                    html.Label("Choose variable"),
                    dcc.Dropdown(
                        options = ["var0","var1"],
                        value = "var0",
                        id = "cross_model_matrix_0_var"
                    ),
                    dcc.RadioItems(['all', 'only constraint compliant', 'only constraint not compliant'], 'all', id="cross_model_matrix_0_constraint",inline=True),
                ], style={'width': '80%', 'display': 'inline-block'}),
                dcc.Graph(id="cross_model_matrix_0")
            ], style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.Div([
                    html.H5("Generated data (by model in x-axis)"),
                    html.Label("Choose variable"),
                    dcc.Dropdown(
                        options = ["var0","var1"],
                        value = "var0",
                        id = "cross_model_matrix_1_var"
                    ),
                    dcc.RadioItems(['all', 'only constraint compliant', 'only constraint not compliant'], 'all', id="cross_model_matrix_1_constraint",inline=True),
                ], style={'width': '80%', 'display': 'inline-block'}),
                dcc.Graph(id="cross_model_matrix_1")
            ], style={'width': '45%', 'display': 'inline-block'}),
        ]),


    ]
)


## Callbacks   
