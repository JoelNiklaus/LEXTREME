import sys
from bertopic import BERTopic
from umap import UMAP
import re

sys.path.append('../code/')
from training_data_handler import TrainingDataHandler
import spacy
from tqdm import tqdm
from textblob_de import TextBlobDE, Word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
import numpy as np
from hdbscan import HDBSCAN
from sentence_transformers import models
from sentence_transformers import SentenceTransformer

tdh = TrainingDataHandler()

language = 'de'
tdh.get_training_data(language=language, affair_text_scope=['zh', 'ch'], affair_attachment_category='all',
                      merge_texts=False, running_mode='experimental')


def replace_specal_characters(text):
    text_new = re.sub(r'ä', r'ae', text)
    text_new = re.sub(r'ü', r'ue', text_new)
    text_new = re.sub(r'ö', r'oe', text_new)
    return text_new


docs = tdh.training_data_df[tdh.training_data_df.split == 'train'].text.unique()
docs = list(docs)
docs_postprocessed = [replace_specal_characters(x) for x in docs]
print(len(docs_postprocessed))
print(len(set(docs_postprocessed)))


# Define custom lemmatizer


class LemmaTokenizer:
    def __init__(self):
        self.nlp = spacy.load("de_dep_news_trf", disable=['ner', 'parser', 'textcat'])
        self.nlp.max_length = 3000000

    def __call__(self, doc):
        blob = TextBlobDE(doc)
        w = TextBlobDE(doc)
        lemmas = w.words.lemmatize()
        words_tags = w.tags
        tags = [x[1] for x in words_tags]
        lemmas_tags = list(zip(lemmas, tags))
        lemmas = [x[0] for x in lemmas_tags if x[1] == 'NN']
        return lemmas
        # return [t.lemma_ for t in tqdm(self.nlp(doc)) if t.pos_ in ['NOUN', 'PROPN']]


vectorizer_model = CountVectorizer(tokenizer=LemmaTokenizer())

# join BERT model and pooling to get the sentence transformer
model = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer', device='mps')

embeddings = model.encode(docs, show_progress_bar=True)

for n in range(5, 20):
    hdbscan_model = HDBSCAN(min_cluster_size=n, metric='euclidean', cluster_selection_method='eom',
                            prediction_data=True)
    topic_model = BERTopic(vectorizer_model=vectorizer_model, hdbscan_model=hdbscan_model)

    topics, probs = topic_model.fit_transform(docs, embeddings)

    indices = [index for index, topic in enumerate(topics) if topic != -1]
    X = embeddings[np.array(indices)]
    labels = [topic for index, topic in enumerate(topics) if topic != -1]

    # Calculate silhouette score
    print(n, ': ', silhouette_score(X, labels))
