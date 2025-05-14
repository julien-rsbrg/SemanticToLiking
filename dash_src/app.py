"""
Main script to run the application

Run in command line: python3 -m dash_src.app
"""

from dash import html, dcc
import dash

import dash_src.data_validation

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        dcc.Store(id="store_cross_exp_selected_indices", data=[]),
        dcc.Store(id="store_sim", data={}),

        html.H1("Exploratory Data Analysis"),
        html.Div(
            [
                html.Div(
                    dcc.Link(f"{page['name']}", href=page["path"]),
                )
                for page in dash.page_registry.values()
            ]
        ),
        html.Hr(),
        dash.page_container,
    ]
)


if __name__ == "__main__":
    dash_src.data_validation.main()

    app.run(debug=True)
