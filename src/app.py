#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
from datetime import date
from datetime import timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from io import BytesIO
import base64

# todays date 
today = date.today()
yesterday = today - timedelta(days = 1)
day_bef_yesterday=today - timedelta(days = 2)
# d1 = yesterday.strftime("%Y-%m-%d")
d1_under=yesterday.strftime("%Y_%m_%d")
# d2=day_bef_yesterday.strftime("%Y-%m-%d")
d2_under=day_bef_yesterday.strftime("%Y_%m_%d")

# Load the data and create necessary variables

rating_curve = pd.read_excel("F://phd_work//cwc_real_time_data//updated_real_time/rating_curve/rating_curve_india_py_new_updated.xlsx", index_col=0)
station_meta = pd.read_excel('station - Copy.xlsx', index_col=0).sort_values('name')
available_st_dis_station = station_meta[station_meta['name'].isin(rating_curve['name'])]

# Create the app
app = dash.Dash(__name__)
server = app.server

# Define the layout
app.layout = html.Div([
    html.H1("Station Data Dashboard"),
    html.H2("This dashbord provides real time discharge and WSE for more details visit :Link"),
    html.Label("Select station:"),
    dcc.Dropdown(
        id='name-dropdown',
        options=[{'label': name, 'value': name} for name in station_meta['name'].unique()],
        value=None,
        clearable=False
    ),
    html.Div(id='output-container', children=[]),
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='wse-plot'),
    dcc.Graph(id='discharge-plot'),
    html.Button('Download Data', id='download-button', disabled=False),
    html.Div(id='print-output')
])

# Define global variables
filtered_df = None
print_output = []

# Define callback functions
@app.callback(
    Output('output-container', 'children'),
    [Input('name-dropdown', 'value')]
)
def update_output(value):
    if value:
        return f"Selected Station: {value}"
    else:
        return ""

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('name-dropdown', 'value')]
)
def update_scatter_plot(value):
    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=station_meta['lat'].round(3),
        lon=station_meta['lon'].round(3),
        mode='markers',
        marker=dict(
            size=6,
            color='red',
        ),
        hovertext=station_meta['name'],
        name='RT-NRT WSE'
    ))

    fig.add_trace(go.Scattermapbox(
        lat=available_st_dis_station['lat'].round(3),
        lon=available_st_dis_station['lon'].round(3),
        mode='markers',
        marker=dict(
            size=8,
            color='green',
        ),
        hovertext=available_st_dis_station['name'],
        name='RT-NRT Discharge'
    ))

    if value:
        selected_meta = station_meta[station_meta['name'] == value]
        fig.add_trace(go.Scattermapbox(
            lat=selected_meta['lat'].round(3),
            lon=selected_meta['lon'].round(3),
            mode='markers',
            marker=dict(
                size=20,
                color='black',
            ),
            hovertext=selected_meta['name'],
            name='Selected station location'
        ))

    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            zoom=3.5,
            center=dict(lat=station_meta['lat'].mean(), lon=station_meta['lon'].mean())
        ),
        legend=dict(
            title='',
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.7)'
        ),
        height=500
    )

    fig.update_layout(margin={'r': 500, 't': 50, 'l': 500, 'b': 0})
    return fig

@app.callback(
    [Output('wse-plot', 'figure'), Output('discharge-plot', 'figure'), Output('download-button', 'disabled')],
    [Input('name-dropdown', 'value')]
)
def update_plots(value):
    global filtered_df
    fig_wse = go.Figure()
    fig_discharge = go.Figure()
    disabled = True

    if value:
        file_paths = f"F:\\phd_work\\cwc_real_time_data\\updated_real_time\\data_files\\2020_updated\\data\\{value}2020_to{d1_under}.xlsx"
        file_paths_yes = f"F:\\phd_work\\cwc_real_time_data\\updated_real_time\\data_files\\2020_updated\\data\\{value}2020_to{d2_under}.xlsx"
        coeff = rating_curve[rating_curve['name'] == value]
        
        if coeff.empty:
            print_output.append(f'No rating curve available for {value}.')
            print(f'No rating curve available for {value}.')
            
            try:
                df_plot = pd.read_excel(file_paths, index_col=0)
            except:
                df_plot = pd.read_excel(file_paths_yes, index_col=0)
            wse_std = np.std(df_plot['WSE'])
            threshold = 5
            filtered_df1 = df_plot[abs(df_plot['WSE'] - np.mean(df_plot['WSE'])) < threshold * wse_std]
            
            fig_wse = px.line(filtered_df1, x='time', y='WSE', title='WSE data at ' + df_plot['name'][0],
                              color_discrete_sequence=['black'])

            fig_wse.update_layout(
                plot_bgcolor='rgb(.9, 0.9, 0.9)'
            )
        else:
            print_output.append(f'Stage discharge relation available for {value} with Discharge RMSE: {coeff["RMSE"].values} and R-square: {coeff["Rsquare"].values}')
            print(f'Stage discharge relation available for {value} with Discharge RMSE: {coeff["RMSE"].values} and R-square: {coeff["Rsquare"].values}')
            
            try:
                df_plot = pd.read_excel(file_paths, index_col=0)
            except:
                df_plot = pd.read_excel(file_paths_yes, index_col=0)
#             print_output.append(f'{df_plot["name"][0]} Discharge RMSE: {coeff["RMSE"].values} and R-square: {coeff["Rsquare"].values}')
#             print(f'{df_plot["name"][0]} Discharge RMSE: {coeff["RMSE"].values} and R-square: {coeff["Rsquare"].values}')

            
            
            
            wse_std = np.std(df_plot['WSE'])
            threshold = 5
            filtered_df1 = df_plot[abs(df_plot['WSE'] - np.mean(df_plot['WSE'])) < threshold * wse_std]
            min_wse_value = coeff['min_wse'].values[0]
            filtered_df = filtered_df1[filtered_df1['WSE'] >= min_wse_value]
            discharge = coeff['Coefficient_p1'].values * (filtered_df['WSE']) ** 2 + coeff['Coefficient_p2'].values * (
                        filtered_df['WSE']) + coeff['Coefficient_p3'].values
            filtered_df['discharge'] = discharge

            fig_wse = px.line(filtered_df, x='time', y='WSE', title='WSE data at ' + df_plot['name'][0],
                              color_discrete_sequence=['black'])

            fig_wse.update_layout(
                plot_bgcolor='rgb(.9, 0.9, 0.9)'
            )

            fig_discharge = px.area(filtered_df, x='time', y='discharge', title='Discharge data at ' + df_plot['name'][0])

            disabled = False

            # Print RMSE and R-square values
            print_output.append(f'{df_plot["name"][0]} Discharge RMSE: {coeff["RMSE"].values} and R-square: {coeff["Rsquare"].values}')
            print(f'{df_plot["name"][0]} Discharge RMSE: {coeff["RMSE"].values} and R-square: {coeff["Rsquare"].values}')

    return fig_wse, fig_discharge, disabled

@app.callback(
    Output('download-button', 'n_clicks'),
    [Input('download-button', 'n_clicks')],
    [State('name-dropdown', 'value')]
)
def download_dataframe(n_clicks, value):
    if n_clicks is not None:
        if filtered_df is not None:
            # Save the DataFrame to an XLSX file
            filtered_df.to_excel('data_' + str(value) + '.xlsx', index=False)
            print_output.append(f'Data exported successfully with file name data_{value}.xlsx')
            print(f'Data exported successfully with file name data_{value}.xlsx')

    return None

@app.callback(
    Output('print-output', 'children'),
    [Input('name-dropdown', 'value'), Input('download-button', 'n_clicks')],
    [State('download-button', 'value')]
)
def update_print_output(value, n_clicks, download_button):
    if n_clicks is not None and filtered_df is not None:
        print_output.append('Download button clicked.')

    if value:
        return [html.Pre(output) for output in print_output]

    return None

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




