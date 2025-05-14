import numpy as np

import plotly.graph_objects as go

from dash import html, Input, Output, State, callback, register_page
from dash import dcc

from dash.exceptions import PreventUpdate

import dash_src.utils as utils
import dash_src.data_load as data_load
import dash_src.display as display

from dash_src.configs.main import config

register_page(__name__)

## Data

## Layout

layout = html.Div(
    [
        dcc.Store(id="store_sim_selected",data={}),

        html.H2("Simulation Level Analysis"),
        html.Div(id="sim_display_sim_selected"),
        html.Div("For this page, only the first sample is kept. If you need to change it, go back to page - cross experiment level."),

        html.H4("Simulation time aggregated results"),
        html.H5("Variables to plot"),
        dcc.Dropdown(
            options=[],
            id = 'sim_simulation_evolution_vars',
            persistence = True,
            persistence_type = "memory",
            multi = True
        ),

        html.Div([
            dcc.RangeSlider(
                min=0,
                max=1,
                id="sim_simulation_evolution_time_range",
                tooltip={"placement": "bottom", "always_visible": True})],
            style={
                'padding': '10px 0px'
        }),
        dcc.Graph(id="sim_simulation_evolution"),

        html.Div([
            html.H4("Simulation reproduction"),
            html.H5("Main species of study"),
            dcc.Dropdown(
                id='sim_reproduction_main_species',
                persistence=True,
                persistence_type="memory"
            ),
            html.Div([
                html.H5("Left plot variable"),
                dcc.Dropdown(
                    options=[],
                    value=None,
                    id='sim_reproduction_0_vars',
                    persistence=True,
                    persistence_type="memory"
                ),
                dcc.RadioItems(
                    options=['Without threshold', 'Use threshold'],
                    value='Without threshold',
                    id='sim_reproduction_0_threshold_button',
                    labelStyle={'display': 'inline-block', 'marginTop': '5px'},
                    persistence=True,
                    persistence_type="memory"
                ),
                dcc.Slider(
                    min=0,
                    max=1,
                    id="sim_reproduction_0_threshold_slider",
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ],
            style={'width': '49%', 'display': 'inline-block'}),

            html.Div([
                html.H5("Right plot variable"),
                dcc.Dropdown(
                    options=[],
                    value=None,
                    id='sim_reproduction_1_vars',
                    persistence=True,
                    persistence_type="memory"
                ),
                dcc.RadioItems(
                    options=['Without threshold', 'Use threshold'],
                    value='Without threshold',
                    id='sim_reproduction_1_threshold_button',
                    labelStyle={'display': 'inline-block', 'marginTop': '5px'},
                    persistence=True,
                    persistence_type="memory"
                ),
                dcc.Slider(
                    min=0,
                    max=1,
                    id="sim_reproduction_1_threshold_slider",
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ],
            style={'width': '49%', 'display': 'inline-block'}),
        ]),

        html.H5("Select time step"),
        dcc.Slider(
            min=0,
            max=1,
            id = "sim_time_slider",
            tooltip={"placement": "bottom", "always_visible": True}
        ),

        html.Div(
            [
                html.Div([
                    dcc.Graph(
                        id='sim_reproduction_0'
                    )
                ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
                html.Div([
                    dcc.Graph(
                        id='sim_reproduction_1'
                    ),
                ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
            ]
        ),

        html.Div(
            [
                html.Div([
                    dcc.Graph(
                        id='sim_reproduction_0_distribution'
                    )
                ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
                html.Div([
                    dcc.Graph(
                        id='sim_reproduction_1_distribution'
                    ),
                ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
            ]
        ),

        ## Debug

    ]
)


## Callbacks

@callback(
    Output("sim_display_sim_selected","children"),
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


@callback(
    Output("store_sim_selected","data"),
    Input("store_cross_exp_selected_indices","data")
)
def update_store_sim_selected(general_store_selected_indices,cross_exp_data=cross_exp_aggregated_results)-> str:
    sim_selected = {}
    if len(general_store_selected_indices):
        sim_id = general_store_selected_indices[0]

        exp_name,sim_name = cross_exp_data.loc[sim_id,["exp_name","sim_name"]]
        sim_selected["sim_id"] = sim_id
        sim_selected["exp_name"] = exp_name
        sim_selected["sim_name"] = sim_name

        return sim_selected
    else:
        raise PreventUpdate
    

@callback(
    Output('sim_simulation_evolution_vars',"options"),
    Input('store_sim_selected', 'data'),
    State('sim_simulation_evolution_vars',"options")
)
def update_sim_simulation_evolution_vars_options(store_sim_selected,current_options):
    # assert all simulation have the same features, so no need to charge options twice    
    if len(store_sim_selected) == 0 or len(current_options) != 0:
        raise PreventUpdate
    
    sim_name,exp_name = store_sim_selected['sim_name'],store_sim_selected["exp_name"]
    sim_analysis_evolution_data = cached_load_simulation_evolution_data(exp_name,sim_name)["simulation_analysis"]

    if len(sim_analysis_evolution_data):
        sim_analysis_evolution_data = sim_analysis_evolution_data.drop(columns=[config["time_column_name"]])
        print("-- complete_sim_evolution_data --\n",sim_analysis_evolution_data)
        return sim_analysis_evolution_data.select_dtypes(include=np.number).columns.tolist()
    else: 
        raise PreventUpdate


@callback(
    Output("sim_simulation_evolution_time_range","min"),
    Output("sim_simulation_evolution_time_range","max"),
    Output("sim_simulation_evolution_time_range","step"),
    Output("sim_simulation_evolution_time_range","marks"),
    Input('store_sim_selected', 'data')
)
def update_cross_exp_simulation_evolution_time_range(store_sim_selected):
    if not(len(store_sim_selected)):
        raise PreventUpdate
    
    sim_name,exp_name = store_sim_selected['sim_name'],store_sim_selected["exp_name"]
    sim_analysis_evolution_data = cached_load_simulation_evolution_data(exp_name,sim_name)["simulation_analysis"]

    if len(sim_analysis_evolution_data):
        timeline = sim_analysis_evolution_data[config["time_column_name"]]
        vmin,vmax,step,marks = utils.get_slider_params(timeline,config["ticks_per_range"])
        return (vmin,vmax,step,marks)
    else:
        raise PreventUpdate
    

@callback(
    Output('sim_simulation_evolution', 'figure'),
    Input('sim_simulation_evolution_vars', 'value'),
    Input('store_sim_selected', 'data'),
    Input('sim_simulation_evolution_time_range',"value"))
def display_simulation_evolution(vars_name,store_sim_selected,time_range):
    fig = go.Figure()

    if vars_name is None or len(vars_name) == 0:
        return fig
    
    sim_name,exp_name = store_sim_selected['sim_name'],store_sim_selected["exp_name"]
    sim_analysis_evolution_data = cached_load_simulation_evolution_data(exp_name,sim_name)["simulation_analysis"]
    
    if not(time_range is None):
        sim_analysis_evolution_data = utils.extract_range(sim_analysis_evolution_data,config["time_column_name"],time_range[0],time_range[1])
    if len(sim_analysis_evolution_data):
        fig = display.create_line_scatter(sim_analysis_evolution_data,config["time_column_name"],vars_name,group_name=exp_name+" - "+sim_name,fig=fig)
    else:
        print('Warning: No data found for simulation %s of experiment %s'%(exp_name,sim_name))

    return fig


@callback(
    Output("sim_reproduction_main_species","value"),
    Output("sim_reproduction_main_species","options"),
    Input("store_sim_selected","data")
)
def update_main_species_dropdown(store_sim_selected):
    sim_name,exp_name = store_sim_selected['sim_name'],store_sim_selected["exp_name"]
    possible_species = utils.get_possible_species(config["result_folder_path"],exp_name,sim_name)
    if len(possible_species)>0:
        return possible_species[0],possible_species
    else:
        return None,possible_species
    

@callback(
    Output("sim_time_slider","min"),
    Output("sim_time_slider","max"),
    Output("sim_time_slider","step"),
    Output("sim_time_slider","marks"),
    Input("store_sim_selected","data")
)
def update_sim_time_slider(store_sim_selected):
    if not(len(store_sim_selected)):
        raise PreventUpdate
    
    sim_name,exp_name = store_sim_selected['sim_name'],store_sim_selected["exp_name"]
    sim_analysis_evolution_data = cached_load_simulation_evolution_data(exp_name,sim_name)["simulation_analysis"]

    if len(sim_analysis_evolution_data):
        timeline = sim_analysis_evolution_data[config["time_column_name"]]
        time_min,time_max = timeline.min(),timeline.max()
        step = timeline.diff().min()

        tick_vals = np.linspace(time_min,time_max,num=config["ticks_per_range"],endpoint=True)
        marks = {v:"%.2f"%v for v in tick_vals}
        return time_min,time_max,step,marks
    else:
        raise PreventUpdate
    

@callback(
    Output("sim_reproduction_0_vars","options"),
    Output("sim_reproduction_0_vars","value"),
    Output("sim_reproduction_1_vars","options"),
    Output("sim_reproduction_1_vars","value"),
    Input("sim_reproduction_main_species","value"),
    Input("store_sim_selected","data")
)
def update_sim_reproduction_controls(main_species,store_sim_selected):
    if not(len(store_sim_selected)) or (main_species is None):
        raise PreventUpdate
    
    sim_name,exp_name = store_sim_selected['sim_name'],store_sim_selected["exp_name"]
    species_data = cached_load_simulation_evolution_data(exp_name,sim_name)[main_species]
    
    if len(species_data):
        num_vars = species_data.drop(columns=config["time_column_name"]).select_dtypes(include=np.number).columns.tolist()
        
        if len(num_vars):
            return num_vars,num_vars[0],num_vars,num_vars[0]
        else:
            print("Warning: no numeric variables for agents")
            raise PreventUpdate
    else:
        raise PreventUpdate

@callback(
    Output("sim_reproduction_0_threshold_slider","min"),
    Output("sim_reproduction_0_threshold_slider","max"),
    Output("sim_reproduction_0_threshold_slider","step"),
    Output("sim_reproduction_0_threshold_slider","marks"),
    Output("sim_reproduction_1_threshold_slider","min"),
    Output("sim_reproduction_1_threshold_slider","max"),
    Output("sim_reproduction_1_threshold_slider","step"),
    Output("sim_reproduction_1_threshold_slider","marks"),
    Input("sim_reproduction_main_species","value"),
    Input("sim_reproduction_0_vars","value"),
    Input("sim_reproduction_1_vars","value"),
    Input("store_sim_selected","data")
)
def update_sim_reproduction_threshold_sliders(main_species,repro_0_var_name, repro_1_var_name, store_sim_selected):
    if not(len(store_sim_selected)) or (main_species is None):
        raise PreventUpdate
    
    sim_name,exp_name = store_sim_selected['sim_name'],store_sim_selected["exp_name"]
    species_data = cached_load_simulation_evolution_data(exp_name,sim_name)[main_species]

    if len(species_data) == 0:
        raise PreventUpdate
    
    else:
        t0_values = species_data[repro_0_var_name]
        t0_vmin,t0_vmax,t0_step,t0_marks = utils.get_slider_params(t0_values,config["ticks_per_range"],config["n_potential_values_continuous_slider"])

        t1_values = species_data[repro_1_var_name]
        t1_vmin,t1_vmax,t1_step,t1_marks = utils.get_slider_params(t1_values,config["ticks_per_range"],config["n_potential_values_continuous_slider"])

        return t0_vmin,t0_vmax,t0_step,t0_marks,t1_vmin,t1_vmax,t1_step,t1_marks


def compute_agents_feature(sim_cycle_data,var_name,threshold_value,use_threshold):
    if use_threshold:
        agents_values = (sim_cycle_data[var_name]>threshold_value).to_numpy()
    else:
        agents_values = sim_cycle_data[var_name].values.to_numpy()

    return agents_values

@callback(
    Output("sim_reproduction_0","figure"),
    Output("sim_reproduction_1","figure"),
    Input("sim_reproduction_main_species","value"),
    Input("sim_reproduction_0_vars","value"),
    Input("sim_reproduction_1_vars","value"),
    Input("sim_reproduction_0_threshold_button","value"),
    Input("sim_reproduction_1_threshold_button","value"),
    Input("sim_reproduction_0_threshold_slider","value"),
    Input("sim_reproduction_1_threshold_slider","value"),
    Input("sim_time_slider","value"),
    Input("store_sim_selected","data")
)
def update_sim_reproduction(main_species,repr_0_var_name,repr_1_var_name,repr_0_use_t,repr_1_use_t,repr_0_val_t,repr_1_val_t,time_value,store_sim_selected):
    if not(len(store_sim_selected)) or (main_species is None):
        raise PreventUpdate
    
    sim_name,exp_name = store_sim_selected['sim_name'],store_sim_selected["exp_name"]
    complete_sim_evolution_data = cached_load_simulation_evolution_data(exp_name,sim_name)
    
    fig_0 = go.Figure()
    fig_1 = go.Figure()
    present_species = []

    for species_name in utils.get_possible_species(config["result_folder_path"],exp_name,sim_name):
        extracted_species_data = complete_sim_evolution_data[species_name]
        mask_time = extracted_species_data[config["time_column_name"]] == time_value

        if len(extracted_species_data[mask_time]):
            present_species.append(species_name)

            if species_name == main_species:
                color_0 = extracted_species_data[repr_0_var_name]
                color_1 = extracted_species_data[repr_1_var_name]

                if repr_0_use_t == 'Use threshold':
                    color_0 = (color_0 >= repr_0_val_t).astype(np.float32)

                if repr_1_use_t == 'Use threshold':
                    color_1 = (color_1 >= repr_1_val_t).astype(np.float32)

                fig_0 = display.create_2d_scatter(
                    extracted_species_data[mask_time],
                    config["xloc_var_name"],
                    config["yloc_var_name"],
                    color_0[mask_time],
                    species_name,
                    cmin=color_0.min(),
                    cmax=color_0.max(),
                    color_showscale=True,
                    marker_symbol=config["species_to_symbol"][species_name],
                    color_bar_title=repr_0_var_name,
                    fig=fig_0)
                fig_1 = display.create_2d_scatter(
                    extracted_species_data[mask_time],
                    config["xloc_var_name"],
                    config["yloc_var_name"],
                    color_1[mask_time],
                    species_name,
                    cmin=color_1.min(),
                    cmax=color_1.max(),
                    color_showscale=True,
                    marker_symbol=config["species_to_symbol"][species_name],
                    color_bar_title=repr_1_var_name,
                    fig=fig_1)
            else:
                fig_0 = display.create_2d_scatter(
                    extracted_species_data[mask_time],
                    config["xloc_var_name"],
                    config["yloc_var_name"],
                    color=None,
                    name=species_name,
                    color_showscale=False,
                    marker_symbol=config["species_to_symbol"][species_name],
                    fig=fig_0)
                
                fig_1 = display.create_2d_scatter(
                    extracted_species_data[mask_time],
                    config["xloc_var_name"],
                    config["yloc_var_name"],
                    color=None,
                    name=species_name,
                    color_showscale=False,
                    marker_symbol=config["species_to_symbol"][species_name],
                    fig=fig_1)
        else:
            if not(time_value is None):
                print("Warning: no data for species %s at time step %.2f"%(species_name,time_value))
            else:
                print("Warning: No time value:",time_value)

    if not(time_value is None):
        title = "Simulation reproduction - time %.2f  <br>Present species: %s"%(time_value,str(present_species))
    else:
        title = "Simulation reproduction - time %s  <br>Present species: %s"%(time_value,str(present_species))
    fig_0.update_layout(title=title)
    fig_1.update_layout(title=title)
    return fig_0,fig_1


@callback(
    Output("sim_reproduction_0_distribution","figure"),
    Output("sim_reproduction_1_distribution","figure"),
    Input("sim_reproduction_main_species","value"),
    Input("sim_reproduction_0_vars","value"),
    Input("sim_reproduction_1_vars","value"),
    Input("sim_reproduction_0_threshold_button","value"),
    Input("sim_reproduction_1_threshold_button","value"),
    Input("sim_reproduction_0_threshold_slider","value"),
    Input("sim_reproduction_1_threshold_slider","value"),
    Input("sim_time_slider","value"),
    Input("store_sim_selected","data")
)
def update_sim_reproduction_distribution(main_species,repr_0_var_name,repr_1_var_name,repr_0_use_t,repr_1_use_t,repr_0_val_t,repr_1_val_t,time_value,store_sim_selected):
    if not(len(store_sim_selected)) or (main_species is None):
        raise PreventUpdate
    
    sim_name,exp_name = store_sim_selected['sim_name'],store_sim_selected["exp_name"]
    complete_sim_evolution_data = cached_load_simulation_evolution_data(exp_name,sim_name)
    
    fig_0 = go.Figure()
    fig_1 = go.Figure()

    extracted_species_data = complete_sim_evolution_data[main_species]
    mask_time = extracted_species_data[config["time_column_name"]] == time_value

    if len(extracted_species_data[mask_time]):
        color_0 = extracted_species_data[repr_0_var_name]
        color_1 = extracted_species_data[repr_1_var_name]
        if repr_0_use_t == 'Use threshold':
            color_0 = (color_0 >= repr_0_val_t).astype(np.float32)

        if repr_1_use_t == 'Use threshold':
            color_1 = (color_1 >= repr_1_val_t).astype(np.float32)

        fig_0 = display.create_simple_histo(
            extracted_species_data[mask_time],
            repr_0_var_name)
        fig_1 = display.create_simple_histo(
            extracted_species_data[mask_time],
            repr_1_var_name)
    else:
        if not(time_value is None):
            print("Warning: no data for species %s at time step %.2f"%(main_species,time_value))
        else:
            print("Warning: No time value:",time_value)
    
    if not(time_value is None):
        title_0 = "Distribution %s in species %s - time %.2f"%(repr_0_var_name,main_species,time_value)
        title_1 = "Distribution %s in species %s - time %.2f"%(repr_1_var_name,main_species,time_value)
    else:
        title_0 = "Distribution %s in species %s - time %s"%(repr_0_var_name,main_species,time_value)
        title_1 = "Distribution %s in species %s - time %s"%(repr_1_var_name,main_species,time_value)

    fig_0.update_layout(title=title_0)
    fig_1.update_layout(title=title_1)

    return fig_0,fig_1


