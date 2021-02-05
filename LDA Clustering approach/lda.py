# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import collections
from collections import Counter
import string
import re
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



import pandas as pd
import numpy as np

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.sklearn_api import LdaTransformer

import pyLDAvis.gensim
import pyLDAvis.sklearn

from tqdm.notebook import tqdm

nlp = spacy.load("en_core_web_lg")

# My list of additional stop words.
stop_list = []

# Updates spaCy's default stop words list with my additional words.
nlp.Defaults.stop_words.update(stop_list)

# Iterates over the words in the stop words list and resets the "is_stop" flag.
for word in STOP_WORDS:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True

datas = pd.read_excel('newData.xlsx')
newest_doc = datas['Use_Case_Description']


def lemmatizer(doc):
    # This takes in a doc of tokens from the NER and lemmatizes them.
    # Pronouns (like "I" and "you" get lemmatized to '-PRON-', so I'm removing those.
    doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    doc = u' '.join(doc)
    return nlp.make_doc(doc)


def remove_stopwords(doc):
    # This will remove stopwords and punctuation.
    # Use token.text to return strings, which we'll need for Gensim.
    doc = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    return doc


# The add_pipe function appends our functions to the default pipeline.
nlp.add_pipe(lemmatizer, name='lemmatizer', after='ner')
nlp.add_pipe(remove_stopwords, name="stopwords", last=True)

doc_list = []
# Iterates through each article in the corpus.
for doc in tqdm(newest_doc):
    # Passes that article through the pipeline and adds to a new list.
    pr = nlp(doc)
    doc_list.append(pr)

# Creates, which is a mapping of word IDs to words.
words = corpora.Dictionary(doc_list)

# Turns each document into a bag of words.
corpus = [words.doc2bow(doc) for doc in doc_list]
punctuations = string.punctuation
stopwords = list(STOP_WORDS)
parser = English()
def spacy_tokenizer(sentence):
    msentence = re.sub('\\n|\\r/g', '', sentence) #removing new lines
    sentence = re.sub('<s>|</s>|<p>|</p>|<@>', '', sentence) #removing start and end tags
    sentence = re.sub('http\S+', '', sentence)   #removing links
    #sentence = re.sub('@\S+', '', sentence) #removes @ sign and anything following like @gmail
    sentence = re.sub('–|--', ' ', sentence) #removes – (long dash) or --
    sentence = re.sub('\/|\”|\“|\(|\)|\:|\;|\’|\‘|\.|!|,|\?|\*|•', ' ', sentence) #removes quotes, slashes, bullet points, parenthesis, etc
    sentence = re.sub("\s+" , " ", sentence) #replaces long spaces or tabs with 1 space
    mytokens = parser(sentence) #adding sentence to lemmatization parser
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ] #punctuations might be redundant because regex is already removing punctions, cant hurt i guess
    mytokens = " ".join([i for i in mytokens])
    return mytokens

tqdm.pandas()
datas["processed_description"] = datas["Use_Case_Description"].progress_apply(spacy_tokenizer)

# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                             id2word=words,
#                                             num_topics=10,
#                                             random_state=2,
#                                             update_every=1,
#                                             passes=10,
#                                             alpha='auto',
#                                             per_word_topics=True)
#
# print('Number of unique tokens: %d' % len(words))
# print('Number of documents: %d' % len(corpus))

NUM_TOPICS = 10
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
datas["processed_description"] = datas['Use_Case_Description'].progress_apply(spacy_tokenizer)

vectorizer = CountVectorizer(ngram_range=(1,2), max_df=0.8)
data_vectorized = vectorizer.fit_transform(datas["processed_description"])
data_lda = lda.fit_transform(data_vectorized)
topics = [[] for _ in range(NUM_TOPICS)]
for item in datas.values:
    x = lda.transform(vectorizer.transform([item[6]]))[0]
    Data = collections.namedtuple("Data", "use_case_num use_case_type topic_num certainty")
    topics[np.argmax(x)].append(item[1])
    print(Data(item[0], item[1], np.argmax(x), np.max(x)))

all_topics = sum((Counter(topic) for topic in topics), Counter())
all_counter = Counter(all_topics)
for topic in topics:
    counter = Counter(topic)

    list = []
    for elem, freq in counter.items():
        list.append((elem, freq/all_counter[elem]))

    print(len(topic))
    print(sorted(list, key=lambda x: float(x[1]), reverse=True), "\n")


visualisation = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')