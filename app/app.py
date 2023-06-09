import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import plotly.io as pio
import plotly.graph_objs as go
import plotly.colors as colors
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from model import read_data, load_bert_model, load_transformed_data, project_to_manifold

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            [dcc.Input(
                id="seed-input",
                type='text',
                placeholder="Type a topic to start",
                debounce=True,
                n_submit=0,
                style={'width': 'auto'}),
                html.Div(id='output_first')]
            , width='auto', style={'margin-left': '15px', 'margin-top': '50px', 'margin-right': '15px'})
    ]),
    dbc.Row([
        dbc.Col(
            [dcc.Dropdown(
                id='my-dropdown',
                placeholder='Hop on a question',
                options=[],
                value=None)]
            , width=10, style={'margin-left': '15px', 'margin-top': '10px', 'margin-right': '15px'})
    ]),
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H5('Answer:'),
                html.Div(id='output')
            ], style={'margin-left': '15px', 'margin-top': '10px', 'height': '800px', 'width': '100%',
                      'overflowY': 'scroll'}),
            width=4
        ),
        dbc.Col(
            dcc.Graph(id='scatter-plot'),
            width=12
        )
    ], style={'display': 'flex', 'flex-wrap': 'nowrap'}),
    dbc.Row([
        html.Div([
            html.P(
                'Data source: Yi Yang, Wen-tau Yih, and Christopher Meek. 2015. '
                'WikiQA: A Challenge Dataset for Open-Domain Question Answering. '
                'In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, '
                'pages 2013–2018', style={'fontSize': 10}),
        ])
    ]),
    dcc.Store(id='past-questions', data=[]),
    dcc.Store(id='prompt', data='placeholder'),
    dcc.Store(id='figure-store')
], fluid=True
)

# Load data and models
data_all = read_data()
model = load_bert_model()
text_mapper = load_transformed_data(data_all, model)
q_mask = data_all['Text_type'] == 'q'
data = data_all[q_mask].copy()
data_answers = data_all[~q_mask].copy()

Text_vec_raw = data['Text'].map(text_mapper)
Text_vec = np.array(Text_vec_raw.tolist())
data['Text_vec'] = Text_vec_raw

embeddings = project_to_manifold(data)
data['x'], data['y'], data['z'] = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]

cluster_estimator = KMeans(n_clusters=24, random_state=0, n_init="auto").fit(data[['x', 'y', 'z']])
data['cluster'] = cluster_estimator.labels_
categories = data['cluster'].unique()
# COLOR MAPPING
# create a list of colors for each point in the trace. %24 is the length of colormap
data['color'] = [colors.qualitative.Alphabet[cat % 24] for cat in data['cluster']]


@app.callback(
    Output('output_first', 'children'),
    Input('seed-input', 'n_submit'),
    State('seed-input', 'value')
)
def cb_render(n_submit, input_value):
    if n_submit > 0:
        return f"You entered: {input_value}"


@app.callback(Output('prompt', 'data'),
              [Input('seed-input', 'value')],
              [Input('my-dropdown', 'value'),
               State('past-questions', 'data')])
def update_output(text_value, dropdown_value, past_q):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'seed-input.value':
        print(f' prompt is "{text_value}"')
        prompt = text_value
        return prompt
    elif ctx.triggered[0]['prop_id'] == 'my-dropdown.value':
        print(f' prompt is "{dropdown_value}"')
        past_q.append(dropdown_value)
        prompt = dropdown_value
        return prompt
    else:
        print(f' DASH NO UPDATE')
        return dash.no_update


@app.callback(
    Output('my-dropdown', 'options'),
    Input('prompt', 'data'),
    State('past-questions', 'data')
)
def update_options(prompt, past_q):
    if prompt is not None:
        vectors = model.encode([prompt])
        distances = pairwise_distances(Text_vec, vectors, metric='cosine', n_jobs=4)
        close_ix = np.argpartition(distances.T, 5)[0]

        choices_raw = list(data.iloc[close_ix[:20]]['Text'].values)
        choices = list(set(choices_raw).difference(set(past_q)))  # remove past questions
        options = [{'label': f'Question: {i}', 'value': i} for i in choices]

        # # Don't know what this one does
        # if prompt is not None:
        #     choices.append({'label': f'Question: {prompt}', 'value': prompt})
    else:
        options = []
    return options


@app.callback(
    Output('output', 'children'),
    [Input('my-dropdown', 'value')]
)
def display_output(selected_value):
    if selected_value is None:
        return ''
    qid = data[data['Text'] == selected_value]['QuestionID'].values[0]
    answers = data_answers[data_answers.QuestionID == qid]['Text']
    max_len = min(len(answers), 5)
    answers = " ".join(answers[:max_len])
    return f'{answers}.'


# Define the callback function
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('my-dropdown', 'value')],
    [State('past-questions', 'data'),
     State('figure-store', 'data')])
def update_scatter_plot(selected_category, past_q, fig_data):
    layout = go.Layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        ),
        showlegend=False,
        uirevision=1  # this is supposed to keep came same between updates. We use "True" here, but it can be any
        # variable whose change should trigger an update. For us, there is none, so it's just True
        # PS: it's not actually working
    )
    fig = pio.from_json(fig_data) if fig_data else go.Figure(layout=layout)
    to_show = data[data.Text.isin(past_q + [selected_category])]
    new_trace = go.Scatter3d(
        x=to_show["x"],
        y=to_show["y"],
        z=to_show['z'],
        mode='markers',
        marker=dict(
            # size=10,
            color=to_show['color']),
        hovertemplate='<b>%{text}</b><extra></extra>',
        text=[title for title in to_show["Text"].values],
    )

    fig.add_trace(new_trace)
    fig.update_layout(height=800, width=800)
    # print(pio.to_json(fig.layout.scene.camera))
    return fig


@app.callback(
    Output('figure-store', 'data'),
    [Input('scatter-plot', 'figure')])
def update_store(fig):
    fig_json = pio.to_json(fig)
    return fig_json


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
