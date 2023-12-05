#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc,html,Input, Output, State
import base64
import os


all_files=os.listdir("dependent/data")

# Load the data and create necessary variables
rating_curve = pd.read_excel("dependent/rating_curve_india_py_new_updated.xlsx", index_col=0)
station_meta = pd.read_excel("dependent/station - Copy.xlsx", index_col=0).sort_values('name')
available_st_dis_station = station_meta[station_meta['name'].isin(rating_curve['name'])]

# Create the app
app = dash.Dash(__name__)
server = app.server

# Define the layout
app.layout = html.Div([
    html.Img(src='data:image/png;base64,{}'.format(base64.b64encode(open('iitblogo_removebg.png', 'rb').read()).decode()), 
             style={'height': '90px', 'width': '300px','float': 'right','margin-top':'10px','margin-right':'30px'}),
    
    html.H1("Real-Time Discharge India", 
            style={
                'text-align': 'center',
#                 'text-decoration': 'underline',
                'font-size':'3em',
                'color': '#cc650a',
                'font-weight': 'bold',
                'line-height': '1.5',
                'margin': '10px',
                'padding': '15px',
#                 'background-color': '#9c9a98', #'#f0f0f0',
                'border': '2px solid #333', 
                
                }
           ),
    
    html.H2(["This dashbord provides real time discharge and WSE for more details visit",
            html.A(": Details", href="https://tinyurl.com/girishiitb", target="_blank")]),
    html.Label("Select station from drop down:",style={'font-size':'1.25em','margin-left':'100px'}),
    dcc.Dropdown(
        id='name-dropdown',
        options=[{'label': name, 'value': name} for name in station_meta['name'].unique()],
        value=None,
        clearable=False,
        style={'position': 'relative', 'left': '5px', 'top': '0px', 'width':'50%', 'zIndex': 1000}  # Adjust left offset as needed
    ),
    html.Div(style={'height': '20px'}),
    html.Div(id='output-container', children=[]),
    dcc.Graph(id='scatter-plot'), #,style={'border': '2px solid black'}
    dcc.Graph(id='wse-plot'),
    dcc.Graph(id='discharge-plot'),
    dcc.Download(id="download-data"),
    html.Div(style={'height': '20px'}),
    html.Button('Download Data', id='download-button', disabled=False, style={'font-size': '18px', 'padding': '5px','margin': 'auto', 'display': 'block'}),
    html.Div(style={'height': '20px'}),
    html.Div(id='print-output',style={'font-size': '18px'}),
],
    style={'backgroundColor': '#f0edfc'}  # Set the background color here

)

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
            style='carto-positron', #'white-bg': White background.
                                    # 'open-street-map': Basic street map.
                                    # 'carto-positron': A clean, light style.
                                    # 'carto-darkmatter': A dark, minimalistic style.

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
        height=550
    )

    fig.update_layout(margin={'r': 650, 't': 50, 'l': 650, 'b': 0})

#     fig.update_layout(
#         margin=dict(r=500, t=50, l=500, b=0),
#         paper_bgcolor="black",  # Set background color of the entire figure
#         plot_bgcolor="white",  # Set background color of the plot area
#         showlegend=False,  # Hide legend for a cleaner look (optional)
#     )

#     fig.update_layout(margin=dict(r=10, l=10, t=10, b=10), width=0.5)

    return fig

@app.callback(
    [Output('wse-plot', 'figure'), Output('discharge-plot', 'figure'), Output('download-button', 'disabled')],
    [Input('name-dropdown', 'value')]
)
def update_plots(value):
    global filtered_df, print_output
    fig_wse = go.Figure()
    fig_discharge = go.Figure()
    disabled = True 

    if value:

        prefix=value+'2020_to'
        matching_files=[filename for filename in all_files if filename.startswith(prefix)]
        file_paths = os.path.join(f"dependent/data/",matching_files[0])

        coeff = rating_curve[rating_curve['name'] == value]
        print_output=[]
        
        if coeff.empty:
            print_output.append(f'No rating curve available for {value}.')
            print(f'No rating curve available for {value}.')
            
            try:
                df_plot = pd.read_excel(file_paths, index_col=0)
            except:
                df_plot = pd.read_excel(file_paths, index_col=0)
            wse_std = np.std(df_plot['WSE'])
            threshold = 5
            filtered_df1 = df_plot[abs(df_plot['WSE'] - np.mean(df_plot['WSE'])) < threshold * wse_std]
            
            filtered_df = filtered_df1 # here if rc is not available it will only save wse
            
            fig_wse = px.line(filtered_df1, x='time', y='WSE', title=f"<b>WSE data at : {df_plot['name'][0]}</b>",
                              color_discrete_sequence=['black'])

            fig_wse.update_layout(
                plot_bgcolor='white',
                xaxis=dict(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgrey',
                    title_font=dict(size=20, family='Arial', color='black'),
                    tickfont=dict(size=16)
                ),
                yaxis=dict(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgrey',
                    title='WSE (m)',
                    title_font=dict(size=20, family='Arial', color='black'),
                    tickfont=dict(size=16)
                )
            )
            
            disabled = False
            
        else:
            print_output.append(f'Stage discharge relation available for {value} with Discharge RMSE: {coeff["RMSE"].values} and R-square: {coeff["Rsquare"].values}')
            print(f'Stage discharge relation available for {value} with Discharge RMSE: {coeff["RMSE"].values} and R-square: {coeff["Rsquare"].values}')
            
            try:
                df_plot = pd.read_excel(file_paths, index_col=0)
            except:
                df_plot = pd.read_excel(file_paths_yes, index_col=0)
            
            wse_std = np.std(df_plot['WSE'])
            threshold = 5
            filtered_df1 = df_plot[abs(df_plot['WSE'] - np.mean(df_plot['WSE'])) < threshold * wse_std]
            min_wse_value = coeff['min_wse'].values[0]
            filtered_df = filtered_df1[filtered_df1['WSE'] >= min_wse_value]
            discharge = coeff['Coefficient_p1'].values * (filtered_df['WSE']) ** 2 + coeff['Coefficient_p2'].values * (
                        filtered_df['WSE']) + coeff['Coefficient_p3'].values
            filtered_df['discharge'] = discharge
            
            fig_wse = px.line(filtered_df, x='time', y='WSE', title=f"<b>WSE data at : {df_plot['name'][0]}</b>",
                              color_discrete_sequence=['black'])
            
            fig_wse.update_layout(
                plot_bgcolor='white',
                xaxis=dict(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgrey',
                    title_font=dict(size=20, family='Arial', color='black'),
                    tickfont=dict(size=16)
                ),
                yaxis=dict(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgrey',
                    title='WSE (m)',
                    title_font=dict(size=20, family='Arial', color='black'),
                    tickfont=dict(size=16)
                )
            )

            fig_discharge = px.area(filtered_df, x='time', y='discharge', title=f"<b>Discharge data at : {df_plot['name'][0]}</b>")

            fig_discharge.update_layout(
                plot_bgcolor='white',
                xaxis=dict(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgrey',
                    title_font=dict(size=20, family='Arial', color='black'),
                    tickfont=dict(size=16)
                ),
                yaxis=dict(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgrey',
                    title='Discharge (cumecs)',
                    title_font=dict(size=20, family='Arial', color='black'),
                    tickfont=dict(size=16)
                )
            )
            
            disabled = False

            # Print RMSE and R-square values
            print_output.append(f'{df_plot["name"][0]} Discharge RMSE: {coeff["RMSE"].values} and R-square: {coeff["Rsquare"].values}')
            print(f'{df_plot["name"][0]} Discharge RMSE: {coeff["RMSE"].values} and R-square: {coeff["Rsquare"].values}')            
    return fig_wse, fig_discharge, disabled

@app.callback(
    Output('download-data', 'data'),
    [Input('download-button', 'n_clicks')],
    [State('name-dropdown', 'value')]
)

def download_data(n_clicks, value):
    if n_clicks and filtered_df is not None:
            print_output.append(f'Data exported successfully for {value} ')
            print(f'Data exported successfully for {value} ')
            return dcc.send_data_frame(filtered_df.to_excel,f"data_{value}.xlsx")
    return None


@app.callback(
    Output('print-output', 'children'),
    [Input('name-dropdown', 'value'), Input('download-button', 'n_clicks')],
    [State('download-button', 'value')]
)
def update_print_output(value, n_clicks, download_button):
    if value:
        return [html.Pre(output) for output in print_output]
    return None


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
