import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import sys

PARENT_DIR = os.path.dirname(sys.path[0]) if os.path.isfile(sys.path[0]) else sys.path[0]


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
