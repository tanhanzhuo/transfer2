from bertopic import BERTopic
from transformers.pipelines import pipeline
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']

# embedding_model = SentenceTransformer("all-mpnet-base-v2")
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = pipeline("feature-extraction", model="princeton-nlp/sup-simcse-roberta-base", device=0)
topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
topics, probs = topic_model.fit_transform(docs)
print(topic_model.get_topic_info())
