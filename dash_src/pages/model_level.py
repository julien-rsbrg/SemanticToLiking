import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from dash import dcc, html, Input, Output, State, callback, register_page, dash_table
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

import dash_src.display as display
import dash_src.utils as utils
import dash_src.data_load as data_load

from dash_src.configs.main import config

register_page(__name__, path='/model_level')

## Data

## Data

CONFIG = {
    "src_results_path":"src/data_generation/examples",
}
RAW_PATH = os.path.join(CONFIG["src_results_path"],"raw")
PROCESSED_PATH = os.path.join(CONFIG["src_results_path"],"processed")

id_to_models_info, all_studies_summaries = data_load.load_data(processed_path=PROCESSED_PATH)

################################################################################################

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
        dcc.Store("model_store_mask_treatment",data=[]),
        dcc.Store("model_store_vars",data=[]),
        dcc.Store("model_store_numeric_vars",data=[]),

        html.H2("Model Level"),

        ### 

        html.H3("Model selection"),
        
        dcc.Dropdown(
            options=[id_to_models_info[i]["model_id"] for i in id_to_models_info.keys()],
            value=id_to_models_info[0]["model_id"],
            id='model_selected',
            multi=False
        ),

        html.H5("Model information:"),
        dcc.Markdown("...model info...",id="model_info_report",style={"width":"80%"}),

        ### 


        html.H3("Parcoords"),
        html.Label("Choose the variables"),
        dcc.Dropdown(
            options=all_studies_summaries.columns.tolist(),
            value=all_studies_summaries.columns.tolist()[:2],
            id='model_parcoords_vars',
            multi=True
        ),

        html.Label("Choose the color variable"),
        dcc.Dropdown(
            options=all_studies_summaries.select_dtypes(include=np.number).columns.tolist(),
            value=all_studies_summaries.select_dtypes(include=np.number).columns.tolist()[0],
            id='model_parcoords_color'
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

        ###
        
        html.H3("Plots"),

        html.H4("Pair plot"),
        html.Label("Choose variables for the axes of the matrix"),

        dcc.Dropdown(
            id = "model_pairplot_dropdown",
            multi=True
        ),
        html.Label("Choose color variable"),
        dcc.Dropdown(
            id = "model_pairplot_dropdown_color",
            multi=False
        ),

        dcc.Graph(id="model_pairplot"),

        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Choose var x"),
                        dcc.Dropdown(
                            id = "model_scatter_0_var_x"
                        )
                    ], style={'width': '45%', 'display': 'inline-block'}),
                    html.Div([
                        html.Label("Choose var y"),
                        dcc.Dropdown(
                            id = "model_scatter_0_var_y"
                        )
                    ], style={'width': '45%', 'display': 'inline-block'}),
                ]),
                dcc.RadioItems(['mix', 'use constraints'], 'mix', id="model_scatter_0_constraints",inline=True),
                dcc.Graph(id="model_scatter_0")
            ], style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Choose var x"),
                        dcc.Dropdown(
                            id = "model_scatter_1_var_x"
                        )
                    ], style={'width': '45%', 'display': 'inline-block'}),
                    html.Div([
                        html.Label("Choose var y"),
                        dcc.Dropdown(
                            id = "model_scatter_1_var_y"
                        )
                    ], style={'width': '45%', 'display': 'inline-block'}),
                ]),
                dcc.RadioItems(['mix', 'use constraints'], 'mix', id="model_scatter_1_constraints",inline=True),
                dcc.Graph(id="model_scatter_1")
            ], style={'width': '45%', 'display': 'inline-block'}),
        ]),
        


        ### 

        html.H3("Inference analysis"),

        html.Label("Write your R-like formula\n(from statsmodels: https://www.statsmodels.org/stable/index.html)"),
        dcc.Textarea(
                style={'width': '80%'},
                id = "model_inference_formula"
        ),
        dcc.RadioItems(["keep everything", "keep only constraint compliants", "keep only not constraint compliants"], 'keep everything', id="model_inference_constraints",inline=True),        
        html.Br(),
        html.Button('Run OLS model', id='button_run_ols', n_clicks=0),
        dcc.Markdown("blabla",id="global_evaluation_report",style={"width":"80%"}),


        ### Debug

    ]
)


## Callbacks
@callback(
    Output("model_info_report","children"),
    Input("model_selected","value"),

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

# options update
@callback(
    Output("model_store_vars","data"),
    Output("model_store_numeric_vars","data"),
    Input("model_selected","value")
)
def update_model_store_vars(model_selected, group_by="model_name", data=all_studies_summaries):
    data, _ = utils.restrict_data(data,group_by,groups_kept=[model_selected],mask_treatment=np.ones(len(data),dtype=bool))
    mask_not_empty = (~data.isna()).sum(axis=0) > 0
    kept_vars = data.columns.to_numpy()[mask_not_empty]
    kept_vars = kept_vars.tolist()

    assert len(kept_vars) > 0, "No data for that model" 

    mask_numeric_not_empty = (~data.select_dtypes(include=np.number).isna()).sum(axis=0) > 0
    kept_numeric_vars = data.select_dtypes(include=np.number).columns.to_numpy()[mask_numeric_not_empty]
    kept_numeric_vars = kept_numeric_vars.tolist()

    assert len(kept_numeric_vars) > 0, "No numeric data for that model" 

    return kept_vars, kept_numeric_vars

@callback(
    Output("model_parcoords_vars","options"),
    Output("model_parcoords_vars","value"),
    Input("model_store_vars","data")
)
def update_model_parcoords_vars(model_options_vars):
    return model_options_vars, model_options_vars[:2]

@callback(
    Output("model_parcoords_color","options"),
    Output("model_parcoords_color","value"),
    Input("model_store_numeric_vars","data")
)
def update_model_parcoords_color(model_options_numeric_vars):
    return model_options_numeric_vars, model_options_numeric_vars[0]

@callback(
    Output("model_pairplot_dropdown","options"),
    Output("model_pairplot_dropdown","value"),
    Input("model_store_vars","data")
)
def update_model_pairplot_dropdown(model_options_vars):
    return model_options_vars, model_options_vars[:2]


@callback(
    Output("model_pairplot_dropdown_color","options"),
    Output("model_pairplot_dropdown_color","value"),
    Input("model_store_numeric_vars","data")
)
def update_model_pairplot_dropdown_color(model_options_numeric_vars):
    _model_options_numeric_vars = model_options_numeric_vars + [None]
    return _model_options_numeric_vars, None


@callback(
    Output("model_scatter_0_var_x","options"),
    Output("model_scatter_0_var_x","value"),
    Input("model_store_vars","data")
)
def update_model_scatter_0_var_x(model_options_vars):
    return model_options_vars, model_options_vars[0]

@callback(
    Output("model_scatter_0_var_y","options"),
    Output("model_scatter_0_var_y","value"),
    Input("model_store_vars","data")
)
def update_model_scatter_0_var_y(model_options_vars):
    return model_options_vars, model_options_vars[0]


@callback(
    Output("model_scatter_1_var_x","options"),
    Output("model_scatter_1_var_x","value"),
    Input("model_store_vars","data")
)
def update_model_scatter_1_var_x(model_options_vars):
    return model_options_vars, model_options_vars[0]

@callback(
    Output("model_scatter_1_var_y","options"),
    Output("model_scatter_1_var_y","value"),
    Input("model_store_vars","data")
)
def update_model_scatter_1_var_y(model_options_vars):
    return model_options_vars, model_options_vars[0]


# parcoords

@callback(
    Output("model_store_mask_treatment","data"),
    Input("model_parcoords","restyleData"),
    Input("model_parcoords","figure")
)
def update_model_mask_treatment(restyle_data,fig_parcoords,data=all_studies_summaries):
    mask_treatment, _ = utils.get_mask_treatment_from_parcoords(fig_parcoords,data)
    return mask_treatment

@callback(
    Output("model_histo_parcoords","figure"),
    Input("model_parcoords_color","value"),
    Input("model_store_mask_treatment","data"),
    Input("model_selected","value")
)
def update_model_only_histo(color_var_name,mask_treatment,model_selected,group_by="model_name",data=all_studies_summaries):
    data, mask_treatment = utils.restrict_data(data,group_by,groups_kept=[model_selected],mask_treatment=np.array(mask_treatment))
    return display.create_parcoords_only_histo(data,color_var_name,mask_treatment)


@callback(
    Output("model_parcoords","figure"),
    Input("model_parcoords_vars","value"),
    Input("model_parcoords_color","value"),
    Input("model_selected","value")
)
def update_model_parcoords(vars_name,color_var_name,model_selected,group_by="model_name",data=all_studies_summaries):
    data, _ = utils.restrict_data(data,group_by,groups_kept=[model_selected],mask_treatment=np.ones(len(data),dtype=bool))
    return display.create_parcoords(data,vars=vars_name,color_var=color_var_name)

# Plots
@callback(
    Output("model_pairplot","figure"),
    Input("model_pairplot_dropdown","value"),
    Input("model_pairplot_dropdown_color","value"),
    Input("model_store_mask_treatment","data"),
    Input("model_selected","value")
)
def update_pairplot(var_names,color_var_name,mask_treatment,model_selected,group_by="model_name",data=all_studies_summaries):
    data, mask_treatment = utils.restrict_data(data,group_by,groups_kept=[model_selected],mask_treatment=np.array(mask_treatment))
    data = data[mask_treatment]
    return display.plot_pairplot_matrix(
        data = data, 
        var_names = var_names, 
        color_var_name = color_var_name, 
        text_var_name = color_var_name)


@callback(
    Output("model_scatter_0","figure"),
    Input("model_scatter_0_var_x","value"),
    Input("model_scatter_0_var_y","value"),
    Input("model_scatter_0_constraints","value"),
    Input("model_store_mask_treatment","data"),
    Input("model_selected","value")
)
def update_model_scatter_0(x_var,y_var,constraints,mask_treatment,model_selected,group_by="model_name",data=all_studies_summaries):
    data, mask_treatment = utils.restrict_data(data,group_by,groups_kept=[model_selected],mask_treatment=np.array(mask_treatment))
    return display.create_pairplot(
        df = data, 
        x_var = x_var,
        y_var = y_var, 
        mask_treatment = None if constraints != "use constraints" else mask_treatment,
        group_by=None
        )


@callback(
    Output("model_scatter_1","figure"),
    Input("model_scatter_1_var_x","value"),
    Input("model_scatter_1_var_y","value"),
    Input("model_scatter_1_constraints","value"),
    Input("model_store_mask_treatment","data"),
    Input("model_selected","value")
)
def update_model_scatter_1(x_var,y_var,constraints,mask_treatment,model_selected,group_by="model_name",data=all_studies_summaries):
    data, mask_treatment = utils.restrict_data(data,group_by,groups_kept=[model_selected],mask_treatment=np.array(mask_treatment))
    return display.create_pairplot(
        df = data, 
        x_var = x_var,
        y_var = y_var, 
        mask_treatment = None if constraints != "use constraints" else mask_treatment,
        group_by=None
        )

# inference analysis
@callback(
    Output("global_evaluation_report","children"),
    Input("button_run_ols","n_clicks"),
    State("model_inference_constraints","value"),
    State("model_inference_formula","value"),
    Input("model_store_mask_treatment","data"),
    State("model_selected","value"),
    prevent_initial_call=True
)
def run_inference(n_clicks,constraints,formula,mask_treatment,model_selected,group_by="model_name",data=all_studies_summaries):
    if formula is None or formula == "":
        return ""

    data, mask_treatment = utils.restrict_data(data,group_by,groups_kept=[model_selected],mask_treatment=np.array(mask_treatment))
    if constraints == "keep only constraint compliants":
        data = data[mask_treatment]
    elif constraints == "keep only not constraint compliants":
        data = data[~mask_treatment]
    
    if len(data) == 0:
        return "No data"

    results = smf.ols(formula, data=data).fit()
    print("== results.summary() ==\n\n",results.summary(),"\n\n == end summary ==")

    # TODO: improve based on this
    # text = results.summary().__repr__()
    # split_text = text.split("\n")
    # _text = ""
    
    # for m in split_text:
    #    _text += m.replace(" ","-") + "\n\n"

    return f"see terminal for:\n\nconstraints: {constraints}\n\nformula: {formula}"