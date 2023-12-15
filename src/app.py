from datetime import datetime, timedelta
import plotly.graph_objects as go
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import time
import plotly.express as px
import requests
import json
import os
from statsmodels.tsa.api import AutoReg
import unicodedata
from prophet import Prophet

# Funciones ficticias para simular datos
def get_temperatures(cities, date_range):
    dates = pd.date_range(date_range[0], date_range[1])
    data = np.random.randint(-10, 30, size=(len(dates), len(cities)))
    df = pd.DataFrame(data, columns=cities, index=dates)
    return df

def calculate_moving_average(dataframe):
    return dataframe.rolling(window=7).mean()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server


########################## CREACION DEL DATAFRAME #################################################################


# Obtener datos de la API
#url = 'https://covid19.secuoyas.io/api/v1/es/ccaa/'
#response = requests.get(url)
#data = response.json()

ruta_json = 'response.json'



#if os.path.exists(ruta_json):
    # Leer el archivo JSON

with open(ruta_json, 'r') as file:
        data = json.load(file)

# Crear un DataFrame vacío para almacenar los datos

print(data)
rows =[]

# Recorrer los datos de la API y construir el DataFrame
for entry in data['timeline']:
    fecha = entry['fecha']  # Obtener la fecha
    regiones = entry['regiones']  # Obtener datos de las regiones
    
    for region in regiones:

        comunidad_autonoma = region['nombreLugar']
        casos_confirmados = region['data']['casosConfirmados']
        casos_hospitalizados = region['data']['casosHospitalizados']
        casos_fallecidos = region['data']['casosFallecidos']
        casos_recuperados = region['data']['casosRecuperados']
        
        # Crear una fila con los datos correspondientes
        row = {
            'Fecha': fecha,
            'Comunidad Autónoma': comunidad_autonoma,
            'Casos Confirmados': casos_confirmados,
            'Casos Hospitalizados': casos_hospitalizados,
            'Casos Fallecidos': casos_fallecidos,
            'Casos Recuperados': casos_recuperados
        }

        rows.append(row)
        
        # Agregar la fila al DataFrame
df = pd.DataFrame(rows)
# Normalizar los nombres de las comunidades autónomas en el DataFrame
df['Comunidad Autónoma'] = df['Comunidad Autónoma'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8'))

# Crear un diccionario de mapeo para corregir los nombres incorrectos
mapeo = {
    'AndalucAa': 'Andalucía',
    'AragA3n': 'Aragón',
    'Castilla y LeA3n': 'Castilla y León',
    'CataluAa': 'Cataluña',
    # Agrega el resto de las correcciones necesarias para las comunidades incorrectas
}

# Aplicar el mapeo al DataFrame para corregir los nombres de las comunidades autónomas
df['Comunidad Autónoma'] = df['Comunidad Autónoma'].map(mapeo).fillna(df['Comunidad Autónoma'])


comunidades_en_df = df['Comunidad Autónoma'].unique()
print(comunidades_en_df)


########################################################################################################################################


# Lista de comunidades autónomas de España
comunidades_autonomas = [
    "Andalucía", "Aragón", "Asturias", "Baleares", "Canarias", "Cantabria", "Castilla-La Mancha", 
    "Castilla y León", "Cataluña", "Extremadura", "Galicia", "Madrid", "Murcia", "Navarra", 
    "País Vasco", "La Rioja", "Valencia"
]

avg = get_temperatures(comunidades_autonomas, ['2001-01-01', '2021-01-01'])
cityAvg = avg.mean(axis=0)

df.index.freq = 'D'


app.layout = html.Div(
    style={'backgroundColor': '#F5F5F5'},
    children=[
        dbc.Container([
            html.H1('Visualización de datos COVID - 19', className='mt-4', style={'textAlign': 'center'}),
            html.Div(id='message-container'),  # Contenedor del mensaje
            html.Div([
                dcc.Dropdown(
                    id='city-dropdown',
                    options=[{'label': c, 'value': c} for c in comunidades_autonomas],
                    value=[],
                    multi=True,
                    clearable=False,
                    style={'maxWidth': '500px', 'margin': '0 auto'}  # Limita el ancho y centra el Dropdown
                )
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            html.Div(
                style={
                    'display': 'grid',
                    'grid-template-columns': '1fr 1fr',  # Ajusta el ancho de las columnas aquí
                    'grid-template-rows': '50% 50%'
                },
                children=[
                    html.Div(
                        style={'grid-column': '1 / span 1', 'grid-row': '1'},
                        children=[
                            dcc.Graph(id='graph1', figure={'data': [], 'layout': {'title': 'Casos confirmados'}}),
                        ]
                    ),
                    html.Div(
                        style={'grid-column': '2 / span 1', 'grid-row': '1'},
                        children=[
                            dcc.Graph(id='graph2', figure={'data': [], 'layout': {'title': 'Casos fallecidos'}}),
                        ]
                    ),
                    html.Div(
                        style={'grid-column': '1 / span 1', 'grid-row': '2'},
                        children=[
                            dcc.Graph(id='graph3', figure={'data': [], 'layout': {'title': 'Casos hospitalizados'}}),
                        ]
                    ),
                    html.Div(
                        style={'grid-column': '2 / span 1', 'grid-row': '2'},
                        children=[
                            dcc.Graph(id='graph4', figure={'data': [], 'layout': {'title': 'Casos recuperados'}}),
                        ]
                    ),
                    html.Div(id='button-container', style={'textAlign': 'center', 'marginTop': '20px'})
                    
                    
                ]
            )
        ]),
        dbc.Container(
            id='tabla-proporciones',
            style={'marginTop': '20px'},  # Ajusta el margen superior según sea necesario
            children=[
                html.Div(
                    style={'grid-column': '1 / span 1', 'grid-row': '3 / span 1'},  # Ajusta la posición
                    children=[
                        html.Table(
                            style={
                                'max-width': '500px',  # Ancho máximo de la tabla
                                'margin': '0 auto',  # Centra horizontalmente
                                'background-color': '#333',
                                'color': 'white',
                                'border-collapse': 'collapse',  # Colapso de bordes para una apariencia más nítida
                            },
                            children=[
                                html.Tr(
                                    children=[
                                            html.Th('Comunidad Autónoma', style={'border': '1px solid white'}),
                                            html.Th('Prop. hospitalizados/confirmados', style={'border': '1px solid white'}),
                                            html.Th('Prop. fallecidos/hospitalizados', style={'border': '1px solid white'}),
                                            html.Th('Prop. recuperados/hospitalizados', style={'border': '1px solid white'}),
                                        ],
                                    style={'text-align': 'center'}
                                ),
                                html.Tbody(id='tabla-body')
                                
                            ]
                        )
                    ]
                )
            ]
        ),
        dbc.Container([
            
            
            html.Div([
                dcc.Checklist(
                    id='mean-checkbox2',
                    options=[{'label': 'Show Predictions', 'value': 'mean'}],
                    value=[]
                )
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.Div(
    style={
        'display': 'grid',
        'grid-template-columns': '1fr 1fr',  # Ajusta el ancho de las columnas aquí
        'grid-template-rows': '1fr 1fr',  # Ajusta la altura de las filas aquí
    },
    children=[
        html.Div(
            style={'grid-column': '1 / span 2', 'grid-row': '1 / span 1'},
            children=[
                dcc.Graph(id='graph12', figure={'data': [], 'layout': {'title': 'Casos confirmados'}}),
            ]
        ),

    ]
)

        ])

    ]
)





@app.callback(
    dash.dependencies.Output('tabla-body', 'children'),
    [dash.dependencies.Input('city-dropdown', 'value')]
)
def update_table(selected_options):
    table_rows = []

    for ccaa in selected_options:
        filtered_data = df[df['Comunidad Autónoma'] == ccaa]
        
        total_confirmados = filtered_data['Casos Confirmados'].sum()
        total_hospitalizados = filtered_data['Casos Hospitalizados'].sum()
        total_fallecidos = filtered_data['Casos Fallecidos'].sum()
        total_recuperados = filtered_data['Casos Recuperados'].sum()
        
        prop_confirmados_hospitalizados = total_hospitalizados /total_confirmados  if total_confirmados != 0 else 0
        prop_hospitalizados_fallecidos = total_fallecidos / total_hospitalizados  if total_hospitalizados != 0 else 0
        prop_hospitalizados_recuperados = total_recuperados /total_hospitalizados  if total_hospitalizados != 0 else 0
        
        # Agregar una fila a la tabla con las proporciones calculadas
        table_rows.append(
            html.Tr([
                html.Td(ccaa),
                html.Td(f'{prop_confirmados_hospitalizados:.2f}', style={'text-align': 'center'}),
                html.Td(f'{prop_hospitalizados_fallecidos:.2f}', style={'text-align': 'center'}),
                html.Td(f'{prop_hospitalizados_recuperados:.2f}', style={'text-align': 'center'})
            ])
        )

    return table_rows






#####################################   callback para los graphs    ##############################################

# Callback para actualizar los gráficos
@app.callback(
    [dash.dependencies.Output('graph1', 'figure'),
     dash.dependencies.Output('graph2', 'figure'),
     dash.dependencies.Output('graph3', 'figure'),
     dash.dependencies.Output('graph4', 'figure'),],
    [dash.dependencies.Input('city-dropdown', 'value')]
)



def update_graphs(selected_options):
    if not selected_options:  
        return {}, {}, {}, {}

    filtered_data = df[df['Comunidad Autónoma'].isin(selected_options)]
    
    data_grouped_confirmed = filtered_data.groupby(['Fecha', 'Comunidad Autónoma'])['Casos Confirmados'].sum().reset_index()
    data_grouped_deaths = filtered_data.groupby(['Fecha', 'Comunidad Autónoma'])['Casos Fallecidos'].sum().reset_index()
    data_grouped_hospitalized = filtered_data.groupby(['Fecha', 'Comunidad Autónoma'])['Casos Hospitalizados'].sum().reset_index()
    data_grouped_recovered = filtered_data.groupby(['Fecha', 'Comunidad Autónoma'])['Casos Recuperados'].sum().reset_index()

    # Generar gráficos
    fig1 = px.line(data_grouped_confirmed, x='Fecha', y='Casos Confirmados', color='Comunidad Autónoma', title='Casos Confirmados por Comunidad Autónoma')
    fig2 = px.line(data_grouped_deaths, x='Fecha', y='Casos Fallecidos', color='Comunidad Autónoma', title='Casos Fallecidos por Comunidad Autónoma')
    fig3 = px.line(data_grouped_hospitalized, x='Fecha', y='Casos Hospitalizados', color='Comunidad Autónoma', title='Casos Hospitalizados por Comunidad Autónoma')
    fig4 = px.line(data_grouped_recovered, x='Fecha', y='Casos Recuperados', color='Comunidad Autónoma', title='Casos Recuperados por Comunidad Autónoma')

    return fig1, fig2, fig3, fig4




#####################################   callback para los graphs    #############################################



# Callback para mostrar mensaje y deshabilitar el Dropdown
@app.callback(
    [dash.dependencies.Output('message-container', 'children'), dash.dependencies.Output('city-dropdown', 'disabled')],
    [dash.dependencies.Input('city-dropdown', 'value')]
)
def handle_dropdown(selected_options):
    if len(selected_options) >= 5:
        
        return html.Div('¡Has alcanzado el máximo de selecciones (5)!'), True
    time.sleep(3) #pausa 3 segundos el mensaje
    return None, False

# Callback para mostrar el botón después de seleccionar 5 opciones
@app.callback(
    dash.dependencies.Output('button-container', 'children'),
    [dash.dependencies.Input('city-dropdown', 'value')]
)
def show_button(selected_options):
    if len(selected_options) >= 5:
        return dbc.Button('Deseleccionar', id='deselect-button', n_clicks=0)
    return None

# Callback para deseleccionar y permitir volver a seleccionar hasta 5 elementos
@app.callback(
    dash.dependencies.Output('city-dropdown', 'value'),
    [dash.dependencies.Input('deselect-button', 'n_clicks')]
)
def reset_selection(n_clicks):
    if n_clicks >= 0:
        return []
    return []  # Valor por defecto después de deseleccionar

@app.callback(
    dash.dependencies.Output('graph12', 'figure'),
    [dash.dependencies.Input('city-dropdown', 'value'),
     dash.dependencies.Input('mean-checkbox2', 'value')]
)
def update_autoregressive_models(selected_options, checkbox_value):
    if not selected_options or 'mean' not in checkbox_value:
        return {}

    fig = go.Figure()

    for selected_country in selected_options:
        filtered_data = df[df['Comunidad Autónoma'] == selected_country]
        filtered_data = filtered_data[['Fecha', 'Casos Confirmados']]
        filtered_data.columns = ['ds', 'y']  # Prophet espera nombres específicos para las columnas
        model = Prophet()
        model.fit(filtered_data)

        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)

        # Agregar datos reales al gráfico
        fig.add_trace(go.Scatter(x=filtered_data['ds'], y=filtered_data['y'],
                                 mode='lines', name=f'Real - {selected_country}'))

        # Agregar predicciones al gráfico con colores diferentes
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                 mode='lines', name=f'Predicción - {selected_country}',
                                 line=dict(dash='dash')))

    fig.update_layout(title='Predicciones de Casos Confirmados por Comunidad Autónoma',
                      xaxis_title='Fecha',
                      yaxis_title='Casos Confirmados')

    return fig




if __name__ == '__main__':
    app.run_server(debug=True)
