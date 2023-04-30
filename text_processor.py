import plotly.express as px
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np
import os
import sys

PARENT_DIR = os.path.dirname(sys.path[0]) if os.path.isfile(sys.path[0]) else sys.path[0]


@st.cache_resource
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


@st.cache_resource()
def load_bert_model(name="all-mpnet-base-v2"):
    """Instantiate a sentence-level DistilBERT model."""
    print('IM RUNNING bert model')

    return SentenceTransformer(name)


@st.cache_resource()
def load_transformed_data(data, _model):
    """Instantiate a sentence-level DistilBERT model."""

    # text_mapper = vectorize_unique_text(_model, data['Text'].unique())
    # with open(os.path.join(PARENT_DIR, 'Text.pickle'), 'wb') as handle:
    #     pickle.dump(text_mapper, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(PARENT_DIR, 'Text.pickle'), 'rb') as handle:
        text_mapper = pickle.load(handle)

    return text_mapper


@st.cache_resource()
def project_to_manifold(_df):
    print('IM RUNNING manifold')
    res = TSNE(n_components=3, learning_rate='auto', init='random', early_exaggeration=18, random_state=42,
               perplexity=45, metric='cosine', n_jobs=8) \
        .fit_transform(np.array(_df['Text_vec'].to_list()))
    return res


def main():
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
    centroids = data[['cluster', 'Text_vec']].groupby('cluster').mean()['Text_vec'].to_frame()
    centroids_xyz = data[['cluster', 'x', 'y', 'z']].groupby('cluster').mean()[['x', 'y', 'z']]

    doc_popularity = data[['DocumentTitle', 'QuestionID']] \
        .groupby(by=['DocumentTitle']) \
        .nunique() \
        .sort_values('QuestionID').rename({'QuestionID': 'popularity'}, axis=1).reset_index()
    data2 = data.merge(doc_popularity, on='DocumentTitle').copy()

    if 'unlocked_q' not in st.session_state:
        st.session_state['unlocked_q'] = []

    start = st.text_input('What question do you want to start from?', key='start_q')

    # def show_qa(data, data_answers):
    #     st.write(f'Question: {st.session_state["current_q"]}')
    #     qid = data[data['Text'] == st.session_state["current_q"]]['QuestionID'].values[0]
    #     answers = data_answers[data_answers.QuestionID == qid]['Text']
    #     st.write(f'Answers: {"".join(answers)[:300]}')
    #     return

    if start:
        if 'current_q' not in st.session_state:
            st.session_state['current_q'] = st.session_state['start_q']
        vecs = model.encode([st.session_state['current_q']])
        distances = pairwise_distances(Text_vec, vecs, metric='cosine', n_jobs=4)
        close_ix = np.argpartition(distances.T, 5)[0]

        choices_raw = list(data.iloc[close_ix[:20]]['Text'].values)
        # remove the entry from the choices so it doesn't become circular
        if st.session_state['current_q'] in choices_raw:
            choices_raw.remove(st.session_state['current_q'])
        choices = list(choices_raw)

        input_topic = st.selectbox(
            'Hop on to a relevant question',
            choices,
            index=2,
            # on_change=show_qa,
        )
        st.session_state['current_q'] = input_topic

        st.write(f'Question: {st.session_state["current_q"]}')
        qid = data[data['Text'] == input_topic]['QuestionID'].values[0]
        answers = data_answers[data_answers.QuestionID == qid]['Text']
        st.write(f'Answers: {"".join(answers)[:300]}')

        st.write(f"you selected {input_topic}")
        st.session_state['unlocked_q'].append(st.session_state['current_q'])
        st.write(st.session_state['unlocked_q'])

    data2['color'] = data2['cluster'].astype(str)
    # fig = px.scatter_3d(
    #     data2,
    #     x="x",
    #     y="y",
    #     z='z',
    #     size=data2['popularity'],
    #     # color=data2['color'],
    #     # color_discrete_map="identity",
    #     color='color',
    #     # color_discrete_map="identity",
    #     # symbol='Text_type',
    #     hover_name="Text",
    #     hover_data={ 'Text_type': False, 'x': False, 'y': False, 'z': False, 'popularity': False},
    #     # log_x=True,
    #     # opacity=0.5,
    #     size_max=20,
    # )
    # fig.update_layout(height=1000, width=1000)
    # fig.update_layout(hovermode="x")
    # fig.update_traces(mode="markers", hoverlabel_align = 'right')
    #
    # st.plotly_chart(fig, theme="streamlit", use_container_width=True, height=1000)



if __name__ == "__main__":
    main()
