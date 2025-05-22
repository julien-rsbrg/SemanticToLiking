import numpy as np
import pandas as pd

from dash import dcc, html, Input, Output, State, callback, register_page, dash_table
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

import dash_src.display as display
import dash_src.utils as utils
import dash_src.data_load as data_load

from dash_src.configs.main import config

register_page(__name__, path='/model_level')

## Data

cross_exp_aggregated_results = pd.DataFrame({"a":[1,2,3],"b":[5,3,2]})
participants_data = pd.DataFrame({"ID":[1,2],"depression_score":[1,5]})
model_to_participant_results = {
    "bayesian_NN":pd.DataFrame({"ID":[1,2],"MAE":[0.2,0.15]}),
    "3NN":pd.DataFrame({"ID":[1,2],"MAE":[0.3,0.2]}),
    "bayesian_GNN":pd.DataFrame({"ID":[1,2],"MAE":[0.21,0.19]})
}

## Layout

layout = html.Div(
    [   
        html.H2("Model Level"),

        ### 

        html.H3("Model selection"),
        
        dcc.Dropdown(
            options=list(model_to_participant_results.keys()),
            value=list(model_to_participant_results.keys())[0],
            id='model_info_selected',
            multi=False
        ),

        html.H5("Model information:"),
        dcc.Markdown("blabla",id="model_info_report",style={"width":"80%"}),

        ### 
        
        html.H3("Plots"),

        html.H4("Pair plot"),
        html.Label("Choose variables"),

        dcc.Dropdown(
            options = ["var0","var1"],
            value = ["var0"],
            id = "model_pairplot_dropdown",
            multi=True
        ),
        
        dcc.Graph(id="model_pairplot"),

        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Choose var x"),
                        dcc.Dropdown(
                            options = ["var0","var1"],
                            value = "var0",
                            id = "model_scatter_0_var_x"
                        )
                    ], style={'width': '45%', 'display': 'inline-block'}),
                    html.Div([
                        html.Label("Choose var y"),
                        dcc.Dropdown(
                            options = ["var0","var1"],
                            value = "var0",
                            id = "model_scatter_0_var_y"
                        )
                    ], style={'width': '45%', 'display': 'inline-block'}),
                ]),
                dcc.Graph()
            ], style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Choose var x"),
                        dcc.Dropdown(
                            options = ["var0","var1"],
                            value = "var0",
                            id = "model_scatter_1_var_x"
                        )
                    ], style={'width': '45%', 'display': 'inline-block'}),
                    html.Div([
                        html.Label("Choose var y"),
                        dcc.Dropdown(
                            options = ["var0","var1"],
                            value = "var0",
                            id = "model_scatter_1_var_y"
                        )
                    ], style={'width': '45%', 'display': 'inline-block'}),
                ]),
                dcc.Graph()
            ], style={'width': '45%', 'display': 'inline-block'}),
        ]),
        


        ### 

        html.H3("Inference analysis"),

        html.Label("Write your R-like formula\n(from statsmodels: https://www.statsmodels.org/stable/index.html)"),
        dcc.Textarea(
                style={'width': '80%'},
                id = "model_formula"
        ),
        html.Br(),
        html.Button('Run OLS model', id='button_run_ols', n_clicks=0),
        dcc.Markdown("blabla",id="global_evaluation_report",style={"width":"80%"}),


        ### Debug

    ]
)


## Callbacks
