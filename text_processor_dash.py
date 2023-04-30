import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output, State
import random
import plotly.express as px
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import plotly.express as px
import numpy as np
import os
import sys

SEED_INPUT = []
PAST_Q = []

PARENT_DIR = os.path.dirname(sys.path[0]) if os.path.isfile(sys.path[0]) else sys.path[0]
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            [dcc.Input(
                         id="seed-input",
                         type='text',
                         placeholder="What's on your mind?",
                         debounce=True,
                         n_submit=0),
                    html.Div(id='output_first')
            ], width=5, style= {'margin-left':'15px', 'margin-top':'50px', 'margin-right':'15px'})
    ]
    ),
    dbc.Row([
        dbc.Col(
            [dcc.Dropdown(
                         id='my-dropdown',
                        placeholder='Hop on a question',
                         options=[],
                         value=None),
                     html.Div(id='output')]
            , width=10, style={'margin-left':'15px', 'margin-top':'10px', 'margin-right':'15px'})
    ]
    ),
    dbc.Row([
        dbc.Col(
            [dcc.Graph(id='scatter-plot')]
        )
    ]
    )
]
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

    with open(os.path.join(PARENT_DIR, 'Text.pickle'), 'rb') as handle:
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
data['color'] = data['cluster'].astype(str)


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
    if not vals[1]: # of it vals[0] has changed
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
    options = [{'label': f'Option {i}', 'value': i} for i in choices_raw]

    # Don't know what this one does
    if selected_value is not None:
        choices_raw.append({'label': f'Option {selected_value}', 'value': selected_value})
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
    return f'Answer: {answers}.'


# Define the callback function
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('my-dropdown', 'value')]
)
def update_scatter_plot(selected_category):
    global PAST_Q

    to_show = data[data.Text.isin(PAST_Q + [selected_category] )]
    fig = px.scatter_3d(
        to_show,
        x="x",
        y="y",
        z='z',
        # size=data2['popularity'],
        # color=data2['color'],
        # color_discrete_map="identity",
        color='color',
        # color_discrete_map="identity",
        # symbol='Text_type',
        hover_name="Text",
        hover_data={ 'Text_type': False, 'x': False, 'y': False, 'z': False},
        # log_x=True,
        # opacity=0.5,
        size_max=20,
    )

    layout = {
        'showlegend': False,
        'xaxis': {
            # 'range': [0.2, 1],
            'showgrid': False, # thin lines in the background
            'zeroline': False, # thick line at x=0
            'visible': False,  # numbers below
        }, # the same for yaxis
        'yaxis': {
            # 'range': [0.2, 1],
            'showgrid': False, # thin lines in the background
            'zeroline': False, # thick line at x=0
            'visible': False,  # numbers below
        }, # the same for yaxis
    }

    # fig.update_layout(height=800, width=800)
    # fig.update_layout(hovermode="x")
    # fig.update_yaxes(automargin=True)
    fig.update_layout(layout)
    fig.update_traces(mode="markers", hoverlabel_align = 'right')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
