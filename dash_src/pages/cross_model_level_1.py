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

        #html.H4("Table Mean-Std"),
        #html.H5("Choose the variables"),
        #dcc.Dropdown(
        #    options=cross_exp_aggregated_mean_results.select_dtypes(include=np.number).columns.tolist(),
        #    value=cross_exp_aggregated_mean_results.select_dtypes(include=np.number).columns.tolist()[:2],
        #    id='cross_exp_vars_table',
        #    multi=True,
        #    persistence=True,
        #    persistence_type="memory"
        #),
        #html.Div(id="cross_exp_mean_std_table"),
        html.H5("Choose the models for comparison"),
        dcc.Dropdown(
            options=list(model_to_participant_results.keys()),
            value=list(model_to_participant_results.keys())[:2],
            id='models_comparison_selected',
            multi=True,
            persistence=True,
            persistence_type="memory"
        ),

        html.H4("Raw distribution of models' performances per participant"),
        dcc.Graph(id="cross_model_raw_distrib"),


        html.H4("Parcoords"),
        html.H5("Choose the variables"),
        dcc.Dropdown(
            options=participants_data.columns.tolist(),
            value=participants_data.columns.tolist()[:2],
            id='cross_model_parcoords_dropdown',
            multi=True,
            persistence=True,
            persistence_type="memory"
        ),

        html.H5("Choose the color variable"),
        dcc.Dropdown(
            options=participants_data.select_dtypes(include=np.number).columns.tolist(),
            value=participants_data.select_dtypes(include=np.number).columns.tolist()[0],
            id='cross_model_color_parcoords_dropdown',
            persistence=True,
            persistence_type="memory"
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
        html.H4("Distributions depending on constraints compliance"),
        dcc.Graph(id="cross_model_group_barchart"),

        html.H4("Distributions depending on participants' features"),
        html.H5("Choose the participant feature for comparison"),
        dcc.Dropdown(
            options=participants_data.columns.tolist(),
            value=participants_data.columns.tolist()[0],
            id='model_comparison_participant_featue_dropdown',
            multi=False,
            persistence=True,
            persistence_type="memory"
        ),

        dcc.Graph(id="cross_model_participant_features_histogram"),

    ]
)


## Callbacks   
@callback(
    Output("cross_model_raw_distrib","figure"),
    Input("models_comparison_selected","value"),

)
def update_model_comparison_raw_distrib(models_selected_value):
    fig = go.Figure()
    for model_name in models_selected_value:
        fig.add_trace(go.Bar(name=model_name,
                             x=participants_data["ID"],
                             y=model_to_participant_results[model_name]["MAE"]))
    
    fig.update_layout(xaxis_title="participant ID",yaxis_title="MAE")
    return fig




@callback(
    Output("cross_model_parcoords","figure"),
    Input("cross_model_parcoords_dropdown","value"),
    Input("cross_model_color_parcoords_dropdown","value")
)
def update_cross_model_parcoords(vars_name,color_var_name):
    return display.create_parcoords(participants_data,vars=vars_name,color_var=color_var_name)


@callback(
    Output("cross_model_store_mask_treatment","data"),
    Input("cross_model_parcoords","restyleData"),
    Input("cross_model_parcoords","figure")
)
def update_cross_model_mask_treatment(restyle_data,fig_parcoords):
    mask_treatment, _ = utils.get_mask_treatment_from_parcoords(fig_parcoords,participants_data)
    return mask_treatment


@callback(
    Output("cross_model_histo_parcoords","figure"),
    Input("cross_model_color_parcoords_dropdown","value"),
    Input("cross_model_store_mask_treatment","data")
)
def update_cross_model_only_histo(color_var_name,mask_treatment):
    return display.create_parcoords_only_histo(participants_data,color_var_name,np.array(mask_treatment))


@callback(
    Output("cross_model_group_barchart","figure"),
    Input("models_comparison_selected","value"),
    Input("cross_model_store_mask_treatment","data")
)
def update_group_barchart(models_selected_value,mask_treatment):
    _participants_data = copy.deepcopy(participants_data)
    _participants_data["constraint compliant"] = mask_treatment

    fig = go.Figure()
    for model_name in models_selected_value:
        # bad TODO: single dataframe bayesian_-perf_name.split("-")[0]
        print("mask_treatment",mask_treatment)
        mean_perf_constraint_compliant = model_to_participant_results[model_name][mask_treatment]["MAE"].mean()
        mean_perf_constraint_not_compliant = model_to_participant_results[model_name][~np.array(mask_treatment)]["MAE"].mean()
        print(mean_perf_constraint_compliant,mean_perf_constraint_not_compliant)
        fig.add_trace(go.Bar(name=model_name,
                             x=["constraint compliant","constraint not compliant"],
                             y=[mean_perf_constraint_compliant,mean_perf_constraint_not_compliant]))
    
    fig.update_layout(xaxis_title="participant ID",yaxis_title="MAE")
    return fig



@callback(
    Output("cross_model_participant_features_histogram","figure"),
    Input("models_comparison_selected","value"),
)
def update_model_comparison_participant_features_histogram(models_selected_value):
    return go.Figure()