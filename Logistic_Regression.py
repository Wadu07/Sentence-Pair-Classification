import json

with open('train.json', 'r', encoding='utf8') as file:
    train = json.load(file)

with open('validation.json', 'r', encoding='utf8') as file:
    valid = json.load(file)

with open('test.json', 'r', encoding='utf8') as file:
    test = json.load(file)

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#stopwords = stopwords.words('romanian')

stemmer = SnowballStemmer('romanian')

def inlocuire_diacritice(propozitie):
    inlocuire = {'ă': 'a', 'î': 'i', 'â': 'a', 'ș': 's', 'ț': 't'}
    for diacritice, inlocuiri in inlocuire.items():
       propozitie = propozitie.replace(diacritice, inlocuiri)
    return propozitie


def eliminare_structuri_html(propozitie):
    propozitie = re.sub('<.*?>+', '', propozitie)
    return propozitie


def normalizare(propozitie):

    #litere mici
    propozitie = propozitie.lower()

    #eliminare diacritice
    propozitie = inlocuire_diacritice(propozitie)

    #eliminare html
    propozitie = eliminare_structuri_html(propozitie)

    #elimminare semne de punctuatie
    propozitie = re.sub(r'[^\w\s]', '', propozitie)

    #Tokenizare
    tokens = nltk.word_tokenize(propozitie)

    #eliminare stopwords
    #filtered_tokens = [word for word in tokens if word not in romanian_stopwords]

    # stemming
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # contopirea tokenilor stemmed
    propozitie = ' '.join(stemmed_tokens)

    return propozitie

def preprocesare(data,nolabel = False):
    labels = []
    guid = []
    propozitie1 = []
    propozitie2 = []
    for inregistrare in data:
        propozitie1.append(normalizare(inregistrare["sentence1"]))
        propozitie2.append(normalizare(inregistrare["sentence2"]))

        if nolabel != True:
            labels.append(inregistrare["label"])

        guid.append(inregistrare["guid"])

    return (propozitie1,propozitie2),labels,guid


preprocessed_train = preprocesare(train)
preprocessed_valid = preprocesare(valid)
preprocessed_test = preprocesare(test,nolabel = True)

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

train_texts = [' '.join(pereche) for pereche in zip(*preprocessed_train[0])]
train_labels = preprocessed_train[1]
valid_texts = [' '.join(pereche) for pereche in zip(*preprocessed_valid[0])]
valid_labels = preprocessed_valid[1]

# Definirea pipeline ului
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('scaler', MaxAbsScaler()),
    ('logreg', LogisticRegression(fit_intercept = False))
])

# parametrii pentru grid search
parametrii_grid = {
    'tfidf__ngram_range': [(1, 1),(1, 2)],
    'logreg__C': [0.2,1,10],
    'logreg__solver': [ 'lbfgs', 'saga'],
    'logreg__penalty': ['l2'],
    'logreg__class_weight': ['balanced'],
    'logreg__multi_class': ['auto'],
    'logreg__max_iter': [1500,3500]
}

# definirirea unui obiect pentru a itera in grid
grid = ParameterGrid(parametrii_grid)

# parcurgerea parametrilor in grid
for parametrii in grid:
    # crearea unui model cu toti parametrii
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=parametrii['tfidf__ngram_range'])),
        ('scaler', MaxAbsScaler()),
        ('logreg', LogisticRegression(C=parametrii['logreg__C'],
                                      solver=parametrii['logreg__solver'],
                                      penalty=parametrii['logreg__penalty'],
                                      class_weight=parametrii['logreg__class_weight'],
                                      multi_class=parametrii['logreg__multi_class'],
                                      max_iter=parametrii['logreg__max_iter']
                                      ))
    ])

    # Antrenarea modelului
    model.fit(train_texts, train_labels)

    # Predictia acestuia
    predicted_labels = model.predict(valid_texts)

    # Afisarea parametrilor curenti cu respectivele scoruri
    print("Current Parameters:", parametrii)
    f1 = f1_score(valid_labels, predicted_labels, average='macro')
    print(f"Macro F1 Score: {f1}")
    print(classification_report(valid_labels, predicted_labels))
    print()

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import cross_val_score


all_train_texts = train_texts + valid_texts
all_train_labels = train_labels + valid_labels

# Crearea unui pipeline cu TF IDF, MaxAbsScaler si Logistic Regression
model = make_pipeline(TfidfVectorizer(ngram_range=(1,2)),
                      MaxAbsScaler(),
                      LogisticRegression(C=0.2,
                                         max_iter=3500,
                                         class_weight='balanced', 
                                         solver = 'saga', 
                                         penalty='l2',
                                         multi_class='auto',
                                         verbose=1,
                                         fit_intercept=False)) 
# Cross-validation
scores = cross_val_score(model, all_train_texts, all_train_labels, cv=5, scoring='f1_macro')
print("Cross-Validation F1 Scores:", scores)

# Antrenarea modelului
model.fit(all_train_texts, all_train_labels)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt


train_texts = [' '.join(pair) for pair in zip(*preprocessed_train[0])]
train_labels = preprocessed_train[1]
valid_texts = [' '.join(pair) for pair in zip(*preprocessed_valid[0])]
valid_labels = preprocessed_valid[1]

model = make_pipeline(TfidfVectorizer(ngram_range=(1,2)),
                      MaxAbsScaler(),
                      LogisticRegression(C=0.2,
                                         max_iter=3500,
                                         class_weight='balanced',
                                         solver='saga',
                                         penalty='l2',
                                         multi_class='auto',
                                         verbose=1,
                                         fit_intercept=False))

# Antrenarea modelului
model.fit(train_texts, train_labels)

# Predict
predicted_labels = model.predict(valid_texts)
print("Classification Report:")
print(classification_report(valid_labels, predicted_labels))

# Matricea de confuzie
confussion_matrix = confusion_matrix(valid_labels, predicted_labels)

plt.figure(figsize=(10,7))
sns.heatmap(confussion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(valid_labels), yticklabels=set(valid_labels))
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

import pandas as pd
import numpy as np

predicted_labels = model.predict(test_texts)

preprocessed_test_array = np.array(preprocessed_test[2]).reshape((-1, 1))
predicted_labels_array = np.array(predicted_labels).reshape((-1, 1))

#Crearea csv ului
data = np.hstack([preprocessed_test_array, predicted_labels_array])
df = pd.DataFrame(data, columns=["guid", "label"])

df.to_csv('LR_saga_qr_final.csv',index=False)