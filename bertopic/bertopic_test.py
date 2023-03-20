from bertopic import BERTopic
from transformers.pipelines import pipeline
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']

embedding_model = pipeline("feature-extraction", model="princeton-nlp/sup-simcse-roberta-base", device=0)
topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
topics, probs = topic_model.fit_transform(docs)
print(topic_model.get_topic_info())
