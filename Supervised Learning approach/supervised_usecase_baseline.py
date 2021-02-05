# Baseline Supervised algorithm


#pip install spacy
#pip install pyldavis
#pip install matplotlib
#pip install sklearn
#pip install plotly
#pip install ipython
#pip install pathlib


import sys
from sys import argv
import re
import numpy as np
import pandas as pd


# spaCy based imports

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy import displacy
#python -m spacy download en_core_web_sm


from tqdm import tqdm
import string
import matplotlib.pyplot as plt
import eli5
import seaborn as sns
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
import pyLDAvis.sklearn
import warnings
warnings.filterwarnings('ignore')

#matplotlib inline

# Plotly based imports for visualization
from plotly import tools
import plotly
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff


nlp = spacy.load('en_core_web_sm')



#opening training file and manipulating data

useCases = pd.read_csv('cases.csv')
newest_doc = useCases['Use_Case_Description']

tempuseCases = pd.read_csv('original_cases.csv')


#tokenizer
parser = English()
def spacy_tokenizer(sentence):
    
    sentence = re.sub('\\n|\\r/g', '', sentence) #removing new lines
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


# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text (tokenizer does this already tho)
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

punctuations = string.punctuation
stopwords = list(STOP_WORDS)

tqdm.pandas()
useCases["processed_description"] = useCases['Use_Case_Description'].progress_apply(spacy_tokenizer)


test = spacy_tokenizer(useCases['Use_Case_Description'][29])

useCasestemp = useCases.drop(columns=['Use_Case_No', 'Use_Case_Type', 'Current_Target', 'Date Added', 'Theme'], axis=1)
useCasestemp.to_csv("Sorta_Processed_data.csv", index=False)


# Creating a vectorizer
#vectorizer = CountVectorizer(tokenizer = spacy_tokenizer,  ngram_range=(1,2)) 

#vvvv this is the right one vvvv
vectorizer = CountVectorizer(ngram_range=(1,2), max_df=0.8)
#classifier = LinearSVC()

bow_vector = CountVectorizer(ngram_range=(1,2), max_df=0.8)
tfidf_vector = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', use_idf=True, ngram_range=(1,2), analyzer='word', token_pattern ="\\b\\w+\\b")

bow_fitted = bow_vector.fit(useCases["processed_description"])
txt_fitted = tfidf_vector.fit(useCases["processed_description"])
tfidf_matrix = tfidf_vector.fit_transform(useCases["processed_description"])



bowfeature_names = np.array(bow_fitted.get_feature_names())
feature_names = np.array(txt_fitted.get_feature_names())

data_vectorized = vectorizer.fit_transform(useCases["processed_description"])
NUM_TOPICS = 10

typeList = list(set(useCases['Use_Case_Type']))
#print(typeList)
print(useCases['Use_Case_Type'].value_counts())

typeList2 = list(set(tempuseCases['Use_Case_Type']))
#print(typeList2)
print(tempuseCases['Use_Case_Type'].value_counts())


X = useCases['Use_Case_Description'] # the features we want to analyze (idk if i need processed ones or normal?)
ylabels = useCases['Use_Case_Type'] # the labels, or answers, we want to test against
#print(X)
#print(ylabels)
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, stratify=ylabels, test_size=0.33, random_state=42)


# Logistic Regression Classifier
classifier = LogisticRegression(random_state=0, multi_class='multinomial')
# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()), ('vectorizer', bow_vector), ('classifier', classifier)])
# model generation
pipe.fit(X_train,y_train)
predicted = pipe.predict(X_test)
# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted, average='micro'))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted, average='micro'))
print("Logistic Regression Count Vectorizor Report: ")
print(metrics.classification_report(y_test, predicted))



# Logistic Regression Classifier with tfidf
classifier = LogisticRegression(random_state=0, multi_class='multinomial')
# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()), ('vectorizer', tfidf_vector), ('classifier', classifier)])
# model generation
pipe.fit(X_train,y_train)
predicted = pipe.predict(X_test)
# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted, average='micro'))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted, average='micro'))
print("Logistic Regression tf-idf Report:")
print(metrics.classification_report(y_test, predicted))




# SVM Classifier
classifier = SVC()
# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()), ('vectorizer', bow_vector), ('classifier', classifier)])
# model generation
pipe.fit(X_train,y_train)
predicted = pipe.predict(X_test)
# Model Accuracy
print("SVM Accuracy:",metrics.accuracy_score(y_test, predicted))
print("SVM Precision:",metrics.precision_score(y_test, predicted, average='micro'))
print("SVM Recall:",metrics.recall_score(y_test, predicted, average='micro'))
print("SVM Count Vectorizor Report: ")
print(metrics.classification_report(y_test, predicted))


# SVM Classifier ifidf
classifier = SVC()
# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()), ('vectorizer', tfidf_vector), ('classifier', classifier)])
# model generation
pipe.fit(X_train,y_train)
predicted = pipe.predict(X_test)
# Model Accuracy
print("SVM Accuracy:",metrics.accuracy_score(y_test, predicted))
print("SVM Precision:",metrics.precision_score(y_test, predicted, average='micro'))
print("SVM Recall:",metrics.recall_score(y_test, predicted, average='micro'))
print("SVM tf-idf Report: ")
print(metrics.classification_report(y_test, predicted))





# NB Classifier
classifier = MultinomialNB(alpha=1)
# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()), ('vectorizer', bow_vector), ('classifier', classifier)])
# model generation
pipe.fit(X_train,y_train)
predicted = pipe.predict(X_test)
# Model Accuracy
print("Multinomial Naive Bayes Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Multinomial Naive Bayes:",metrics.precision_score(y_test, predicted, average='micro'))
print("Multinomial Naive Bayes:",metrics.recall_score(y_test, predicted, average='micro'))
print("Multinomial Naive Bayes Count Vectorizor Report: ")
print(metrics.classification_report(y_test, predicted))


# NB Classifier ifidf
classifier = MultinomialNB(alpha=1)
# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()), ('vectorizer', tfidf_vector), ('classifier', classifier)])
# model generation
pipe.fit(X_train,y_train)
predicted = pipe.predict(X_test)
# Model Accuracy
print("Multinomial Naive Bayes Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Multinomial Naive Bayes Precision:",metrics.precision_score(y_test, predicted, average='micro'))
print("Multinomial Naive Bayes Recall:",metrics.recall_score(y_test, predicted, average='micro'))
print("Multinomial Naive Bayes tf-idf Report: ")
print(metrics.classification_report(y_test, predicted))





# RF Classifier
classifier = RandomForestClassifier()
# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()), ('vectorizer', bow_vector), ('classifier', classifier)])
# model generation
pipe.fit(X_train,y_train)
predicted = pipe.predict(X_test)
# Model Accuracy
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Random Forest:",metrics.precision_score(y_test, predicted, average='micro'))
print("Random Forest:",metrics.recall_score(y_test, predicted, average='micro'))
print("Random Forest Count Vectorizor Report: ")
print(metrics.classification_report(y_test, predicted))


# RF Classifier ifidf
classifier = RandomForestClassifier()
# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()), ('vectorizer', tfidf_vector), ('classifier', classifier)])
# model generation
pipe.fit(X_train,y_train)
predicted = pipe.predict(X_test)
# Model Accuracy
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Random Forest Precision:",metrics.precision_score(y_test, predicted, average='micro'))
print("Random Forest Recall:",metrics.recall_score(y_test, predicted, average='micro'))
print("Random Forest tf-idf Report: ")
print(metrics.classification_report(y_test, predicted))





#plotting
idf = tfidf_vector.idf_

rr = dict(zip(txt_fitted.get_feature_names(), idf))
token_weight = pd.DataFrame.from_dict(rr, orient='index').reset_index()
token_weight.columns=('token','weight')
token_weight = token_weight.sort_values(by='weight', ascending=False)
token_weight 

sns.barplot(x='token', y='weight', data=token_weight[0:25])            
plt.title("Inverse Document Frequency(idf) per token")
fig=plt.gcf()
fig.set_size_inches(10,5)
#plt.show() #commented out for now


#this doesnt do anything rn
eli5.show_weights(classifier, vec=bow_vector, top=10, target_names=typeList)


# Non-Negative Matrix Factorization Model
nmf = NMF(n_components=NUM_TOPICS)
data_nmf = nmf.fit_transform(data_vectorized) 

# Latent Semantic Indexing Model using Truncated SVD
lsi = TruncatedSVD(n_components=100)
data_lsi = lsi.fit_transform(data_vectorized)


# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]]) 


# Keywords for topics clustered by Non-Negative Matrix Factorization
print("NMF Model:")
#selected_topics(nmf, vectorizer)

# Keywords for topics clustered by Latent Semantic Indexing
print("LSI Model:")
#selected_topics(lsi, vectorizer)


visualisation = pyLDAvis.sklearn.prepare(nmf, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(visualisation, 'Lsi_Visualization.html')

#plotting


svd_2d = TruncatedSVD(n_components=2)
data_2d = svd_2d.fit_transform(data_vectorized)

trace = go.Scattergl(
    x = data_2d[:,0],
    y = data_2d[:,1],
    mode = 'markers',
    marker = dict(
        color = '#FFBAD2',
        line = dict(width = 1)
    ),
    text = tfidf_vector.get_feature_names(),
    hovertext = tfidf_vector.get_feature_names(),
    hoverinfo = 'text' 
)
data = [trace]
plotly.offline.plot(data, filename='scatter-mode.html', auto_open=False)



print("done")


