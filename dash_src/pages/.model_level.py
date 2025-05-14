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
participants_data = pd.DataFrame({"ID":[1,2],"depression_score":[1,5]})
model_to_participant_results = {
    "bayesian_NN":pd.DataFrame({"ID":[1,2],"MAE":[0.2,0.15]}),
    "3NN":pd.DataFrame({"ID":[1,2],"MAE":[0.3,0.2]}),
    "bayesian_GNN":pd.DataFrame({"ID":[1,2],"MAE":[0.21,0.19]})
}

## Layout

layout = html.Div(
    [   
        #dcc.Store("cross_exp_store_mask_treatment"),
        #dcc.Store("cross_exp_store_simulation_selected_indices",data=[]),

        html.H2("Model Level"),

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

        html.H4("Parcoords"),
        html.H5("Choose the variables"),
        dcc.Dropdown(
            options=participants_data.columns.tolist(),
            value=participants_data.columns.tolist()[:2],
            id='participants_parcoords',
            multi=True,
            persistence=True,
            persistence_type="memory"
        ),

        html.H5("Choose the color variable"),
        dcc.Dropdown(
            options=participants_data.select_dtypes(include=np.number).columns.tolist(),
            value=participants_data.select_dtypes(include=np.number).columns.tolist()[0],
            id='participants_color_parcoords',
            persistence=True,
            persistence_type="memory"
        ),
        html.Div([
            html.Div([
                dcc.Graph(id="model_histo_parcoords")
                ],
                style={'width': '15%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id="model_parcoords")
                ],
                style={'width': '85%', 'display': 'inline-block'}),
        ]),
        html.H4("Distributions depending on constraints compliance"),
        dcc.Graph(id="model_participant_histogram"),
        dcc.Graph(id="model_group_histogram"),
        dcc.Graph(id="model_group_histogram"),

        html.H4("Distributions depending on constraints compliance"),
        dcc.Graph(id="model_violins"),

        html.Div([
            html.Div([
                html.H4("Pairplot of models' results"),
                html.Div([
                    html.H5("X variable"),
                    dcc.Dropdown(
                        options=cross_exp_aggregated_results.select_dtypes(include=np.number).columns.tolist(),
                        value=cross_exp_aggregated_results.select_dtypes(include=np.number).columns.tolist()[0],
                        id='cross_exp_pairplot_xaxis',
                        persistence=True,
                        persistence_type="memory"
                )],style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                    html.H5("Y variable"),
                    dcc.Dropdown(
                        options=cross_exp_aggregated_results.select_dtypes(include=np.number).columns.tolist(),
                        value=cross_exp_aggregated_results.select_dtypes(include=np.number).columns.tolist()[0],
                        id='cross_exp_pairplot_yaxis',
                        persistence=True,
                        persistence_type="memory"
                )],style={'width': '49%', 'display': 'inline-block'}),
                dcc.RadioItems(
                    options=['Ignore constraints', 'Consider constraints'],
                    value='Ignore constraints',
                    id='cross_exp_pairplot_constraints',
                    labelStyle={'display': 'inline-block', 'marginTop': '5px'},
                    persistence=True,
                    persistence_type="memory"
                )
            ],
            style={'width': '49%', 'display': 'inline-block'}),

            html.Div([
                html.H4("Evolution of variables over time in simulations"),
                dcc.Dropdown(
                    options=[],
                    value=[],
                    id='cross_exp_simulation_evolution_vars',
                    multi=True,
                    persistence=True,
                    persistence_type="memory"
                ),
                html.Div([dcc.RangeSlider(
                    min=0,
                    max=1,
                    id="cross_exp_simulation_evolution_time_range",
                    tooltip={"placement": "bottom", "always_visible": True})],
                style={
                    'padding': '10px 0px'
                }),
            ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
        ], style={
            'padding': '10px 5px'
        }),

        html.Div([
            html.Div([
                dcc.Graph(
                    id='cross_exp_pairplot'
                )
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([
                dcc.Graph(
                    id='cross_exp_simulation_evolution'
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20'}),
        ]),
        
        html.H2("Select samples to further investigate"),
        dcc.Dropdown(
            options=["Investigate constraints compliants samples",
                     "Investigate non constraints compliants samples",
                     "Investigate selected samples in pairplot"],
            value = "Investigate selected samples in pairplot",
            id="cross_exp_store_selected_indices_dropdown",
            persistence=True,
            persistence_type="memory"
        ),
        html.Label(id="cross_exp_display_samples_selected"),


        ### Debug

    ]
)


## Callbacks   
@callback(
    Output("cross_exp_mean_std_table","children"),
    Input("cross_exp_vars_table","value")
)
def update_mean_std_table(vars_name,mean_std_data=cross_exp_aggregated_mean_std_results,mean_data=cross_exp_aggregated_mean_results):
    numeric_columns = set(mean_data.select_dtypes(include=np.number).columns)
    not_numeric_columns = set(mean_data.columns)-numeric_columns
    _vars_name = list(not_numeric_columns)+vars_name

    extracted_data = mean_std_data[_vars_name].sort_values(by=_vars_name[0])
    return dash_table.DataTable(extracted_data.to_dict('records'), [{"name": i, "id": i} for i in extracted_data.columns])

@callback(
    Output("cross_exp_parcoords","figure"),
    Input("cross_exp_vars_parcoords","value"),
    Input("cross_exp_color_parcoords","value")
)
def update_cross_exp_parcoords(vars_name,color_var_name):
    return display.create_parcoords(cross_exp_aggregated_results,vars=vars_name,color_var=color_var_name)


@callback(
    Output("cross_exp_store_mask_treatment","data"),
    Input("cross_exp_parcoords","restyleData"),
    Input("cross_exp_parcoords","figure")
)
def update_cross_exp_mask_treatment(restyle_data,fig_parcoords,data=cross_exp_aggregated_results):
    mask_treatment, _ = utils.get_mask_treatment_from_parcoords(fig_parcoords,data)
    return mask_treatment



@callback(
    Output("cross_exp_histo_parcoords","figure"),
    Input("cross_exp_color_parcoords","value"),
    Input("cross_exp_store_mask_treatment","data")
)
def update_cross_exp_only_histo(color_var_name,mask_treatment,data=cross_exp_aggregated_results):
    return display.create_parcoords_only_histo(data,color_var_name,np.array(mask_treatment))



@callback(
    Output("cross_exp_violins","figure"),
    Input("cross_exp_vars_parcoords","value"),
    Input("cross_exp_color_parcoords","value"),
    Input("cross_exp_store_mask_treatment","data")
)
def update_cross_exp_violins(vars_name,color_var_name,mask_treatment,data=cross_exp_aggregated_results):
    _vars_name = ([color_var_name]+list(set(vars_name)-set([color_var_name])))
    return display.create_violins(data,_vars_name,np.array(mask_treatment))


@callback(
    Output("cross_exp_pairplot","figure"),
    Input("cross_exp_pairplot_xaxis","value"),
    Input("cross_exp_pairplot_yaxis","value"),
    Input("cross_exp_pairplot_constraints","value"),
    Input("cross_exp_store_mask_treatment","data")
)
def update_cross_exp_pairplot(xaxis_var_name,yaxis_var_name,consider_constraints,mask_treatment):
    if consider_constraints == 'Ignore constraints':
        return display.create_pairplot(cross_exp_aggregated_results,xaxis_var_name,yaxis_var_name,mask_treatment=None)
    else:
        return display.create_pairplot(cross_exp_aggregated_results,xaxis_var_name,yaxis_var_name,mask_treatment=np.array(mask_treatment))

@callback(
    Output('cross_exp_store_simulation_selected_indices', 'data'),
    Input('cross_exp_pairplot', 'selectedData'),
    Input('cross_exp_pairplot', 'clickData'))
def store_selected_indices(selected_data,click_data):
    selected_indices = []
    if not(selected_data is None):
        selected_indices += [selected_data["points"][i]["pointIndex"] for i in range(len(selected_data["points"]))]
    elif not(click_data is None):
        selected_indices += [click_data["points"][0]["pointIndex"]]
        
    return selected_indices


@callback(
    Output('cross_exp_simulation_evolution_vars',"options"),
    Input('cross_exp_store_simulation_selected_indices', 'data'),
    State('cross_exp_simulation_evolution_vars',"options")
)
def update_cross_exp_simulation_evolution_vars_options(selected_indices,current_options,cross_exp_data=cross_exp_aggregated_results):
    # assert all simulation have the same features, so no need to charge options twice
    if len(selected_indices)>0 and len(current_options) == 0:
        for row_index in selected_indices:
            sim_name,exp_name = cross_exp_data.loc[row_index,['sim_name','exp_name']].values
            sim_analysis_evolution_data = data_load.load_simulation_evolution_data(exp_name,sim_name)["simulation_analysis"]
            if len(sim_analysis_evolution_data):
                sim_analysis_evolution_data = sim_analysis_evolution_data.drop(columns=[config["time_column_name"]])
                return sim_analysis_evolution_data.select_dtypes(include=np.number).columns.tolist()
    else: 
        raise PreventUpdate

@callback(
    Output("cross_exp_simulation_evolution_time_range","min"),
    Output("cross_exp_simulation_evolution_time_range","max"),
    Output("cross_exp_simulation_evolution_time_range","step"),
    Output("cross_exp_simulation_evolution_time_range","marks"),
    Input('cross_exp_store_simulation_selected_indices', 'data'),
)
def update_cross_exp_simulation_evolution_time_range(selected_indices,cross_exp_data=cross_exp_aggregated_results):
    if len(selected_indices)>0:
        for row_index in selected_indices:
            sim_name,exp_name = cross_exp_data.loc[row_index,['sim_name','exp_name']].values
            sim_analysis_evolution_data = data_load.load_simulation_evolution_data(exp_name,sim_name)["simulation_analysis"]
            if len(sim_analysis_evolution_data):
                timeline = sim_analysis_evolution_data[config["time_column_name"]]
                vmin,vmax,step,marks = utils.get_slider_params(timeline,config["ticks_per_range"])
                return (vmin,vmax,step,marks)
            else:
                raise PreventUpdate
    else: 
        raise PreventUpdate

@callback(
    Output('cross_exp_simulation_evolution', 'figure'),
    Input('cross_exp_simulation_evolution_vars', 'value'),
    Input('cross_exp_store_simulation_selected_indices', 'data'),
    Input('cross_exp_simulation_evolution_time_range',"value"))
def display_simulation_evolution(vars_name,selected_indices,time_range,cross_exp_data=cross_exp_aggregated_results):
    fig = go.Figure()

    if len(vars_name) == 0:
        return fig
    
    for row_index in selected_indices:
        sim_name,exp_name = cross_exp_data.loc[row_index,['sim_name','exp_name']].values
        
        sim_analysis_evolution_data = data_load.load_simulation_evolution_data(exp_name,sim_name)["simulation_analysis"]
        if not(time_range is None):
            sim_analysis_evolution_data = utils.extract_range(sim_analysis_evolution_data,config["time_column_name"],time_range[0],time_range[1])
        if len(sim_analysis_evolution_data):
            fig = display.create_line_scatter(sim_analysis_evolution_data,config["time_column_name"],vars_name,group_name=exp_name+" - "+sim_name,fig=fig)
        else:
            print('Warning: No data found for simulation %s of experiment %s'%(sim_name,exp_name))

    return fig


@callback(
    Output("store_cross_exp_selected_indices","data"),
    Input("cross_exp_store_selected_indices_dropdown","value"),
    Input("cross_exp_store_mask_treatment","data"),
    Input('cross_exp_store_simulation_selected_indices', 'data'),
)
def update_store_exp(dropdown_value:str|int,mask_treatment:list,pairplot_selected_indices:list,cross_exp_data=cross_exp_aggregated_results)-> list:
    """
    Keep the selected experiment in store
    """
    if dropdown_value == "Investigate constraints compliants samples":
        return cross_exp_data[np.array(mask_treatment)].index.tolist()
    elif dropdown_value == "Investigate non constraints compliants samples":
        return cross_exp_data[~np.array(mask_treatment)].index.tolist()
    elif dropdown_value == "Investigate selected samples in pairplot":
        return pairplot_selected_indices
    elif dropdown_value is None:
        raise PreventUpdate


@callback(
    Output("cross_exp_display_samples_selected","children"),
    Input("store_cross_exp_selected_indices","data")
)
def display_exp_selected(general_store_selected_indices:dict,cross_exp_data=cross_exp_aggregated_results)-> str:
    """
    Inform of the selected experiment for further investigation
    """
    if len(general_store_selected_indices):
        message = "Indices ["
        for id in general_store_selected_indices:
            message += str(id)+", "
        message += "] were selected.\n This corresponds to the (experiment_name,simulation_name) tuples:\n ["
        for id in general_store_selected_indices:
            exp_name,sim_name = cross_exp_data.loc[id,["exp_name","sim_name"]]
            message += "("+exp_name+","+sim_name+"),"
        message += "]"
        return message
    else :
        return "No samples selected or in memory."
