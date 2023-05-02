import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output, State
import numpy as np
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import plotly.graph_objs as go
import plotly.colors as colors
import os
import sys

SEED_INPUT = []
PAST_Q = []

PARENT_DIR = os.path.dirname(sys.path[0]) if os.path.isfile(sys.path[0]) else sys.path[0]

layout = go.Layout(
    scene=dict(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        zaxis=dict(showticklabels=False),
    ),
    showlegend=False,
)

fig = go.Figure(data=[], layout=layout)
fig.update_layout(height=800, width=800)

app = dash.Dash(__name__)  # , external_stylesheets=[dbc.themes.CERULEAN])

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
                html.Div(id='output_first')
            ], width='auto', style={'margin-left': '15px', 'margin-top': '50px', 'margin-right': '15px'})
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
                html.Div(id='output',
                         style={'margin-left': '15px', 'height': '300px', 'width': '100%', 'overflowY': 'scroll'})
            ]),
            width=2
        ),
        dbc.Col(
            dcc.Graph(id='scatter-plot', figure=fig),
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
    dcc.Store(id='past-ali-values', data=[])
], fluid=True
)


def read_data():
    """Read the data from local."""
    data = os.path.join(PARENT_DIR, 'data', 'WikiQACorpus', 'WikiQA.tsv')
    d = pd.read_csv(data, delimiter='\t')
    q = d.drop_duplicates(subset='Question').copy()
    q.rename({'Question': 'Text'}, axis=1, inplace=True)
    q['SentenceID'] = None
    q['Text_type'] = 'q'

    s = d.copy()
    s.rename({'Sentence': 'Text'}, axis=1, inplace=True)
    s['Text_type'] = 's'

    concatenated = pd.concat([q, s])
    col_relevant = ['QuestionID', 'Text', 'DocumentID', 'DocumentTitle', 'SentenceID', 'Label', 'Text_type']
    return concatenated[col_relevant].copy()


def vectorize_unique_text(model, list_of_str):
    "returns a dict to map vectors to strings"
    vecs = model.encode(list_of_str)
    return dict(zip(list(list_of_str), vecs))


def load_bert_model(name="all-mpnet-base-v2"):
    """Instantiate a sentence-level DistilBERT model."""
    print('IM RUNNING bert model')

    return SentenceTransformer(name)


def load_transformed_data(data, _model):
    """Instantiate a sentence-level DistilBERT model."""

    # text_mapper = vectorize_unique_text(_model, data['Text'].unique())
    # with open(os.path.join(PARENT_DIR, 'Text.pickle'), 'wb') as handle:
    #     pickle.dump(text_mapper, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(PARENT_DIR, 'data/Text.pickle'), 'rb') as handle:
        text_mapper = pickle.load(handle)

    return text_mapper


def project_to_manifold(_df):
    print('IM RUNNING manifold')
    res = TSNE(n_components=3, learning_rate='auto', init='random', early_exaggeration=18, random_state=42,
               perplexity=45, metric='cosine', n_jobs=8) \
        .fit_transform(np.array(_df['Text_vec'].to_list()))
    return res


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

clusterer = KMeans(n_clusters=24, random_state=0, n_init="auto").fit(data[['x', 'y', 'z']])
data['cluster'] = clusterer.labels_
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
        SEED_INPUT = input_value
        return f"You entered: {input_value}"


@app.callback(
    Output('my-dropdown', 'options'),
    [Input('seed-input', 'value'),
     Input('my-dropdown', 'value')])
def update_options(*vals):
    #######
    # THIS IS WHERE THE CHOICES GET CREATED
    global SEED_INPUT
    global PAST_Q
    if not vals[1]:  # of it vals[0] has changed
        selected_value = vals[0]
        SEED_INPUT.append(vals[0])
    else:
        selected_value = vals[1]
        PAST_Q.append(vals[1])

    if not vals[0] and not vals[1]:
        selected_value = 'placeholder'

    vecs = model.encode([selected_value])
    distances = pairwise_distances(Text_vec, vecs, metric='cosine', n_jobs=4)
    close_ix = np.argpartition(distances.T, 5)[0]

    choices_raw = list(data.iloc[close_ix[:20]]['Text'].values)
    choices = list(set(choices_raw).difference(set(PAST_Q)))  # remove past questions
    options = [{'label': f'Question: {i}', 'value': i} for i in choices]

    # Don't know what this one does
    if selected_value is not None:
        choices.append({'label': f'Question: {selected_value}', 'value': selected_value})
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
    answers = " ".join(answers[:max_len])[:750]
    return f'{answers}.'


# Define the callback function
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('my-dropdown', 'value')]
)
def update_scatter_plot(selected_category):
    global PAST_Q

    to_show = data[data.Text.isin(PAST_Q + [selected_category])]
    new_trace = go.Scatter3d(
        x=to_show["x"],
        y=to_show["y"],
        z=to_show['z'],
        mode='markers',
        marker=dict(
            # size=10,
            color=to_show['color']),
        hovertemplate='<b>%{text}</b><extra></extra>',
        text=[title for title in to_show["Text"].values]  # [title for title in df.Title],
    )
    fig.add_trace(new_trace)
    return fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
