
# Projeto 01 - Classificação de Notícias Falsas

#Puxar os dados#

##Download dos datasets##

#import kagglehub

# Download latest version
#path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

#print("Path to dataset files:", path)#

#pip install pandas matplotlib seaborn wordcloud nltk scikit-learn kaggle

import pandas as pd

# Carregar os datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Adicionar rótulo
fake["label"] = 0
real["label"] = 1

# Concatenar
df = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)

df.head()

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['text'] = df['title'] + " " + df['text']
df['clean_text'] = df['text'].apply(preprocess)

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Distribuição das classes
sns.countplot(x='label', data=df)
plt.title('Distribuição de Notícias')
plt.xticks([0,1], ['Fake', 'Real'])
plt.show()

# Nuvem de palavras
fake_words = ' '.join(df[df.label == 0]['clean_text'])
real_words = ' '.join(df[df.label == 1]['clean_text'])

WordCloud().generate(fake_words).to_image()
WordCloud().generate(real_words).to_image()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Separar dados
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

# Vetorização
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Treinamento
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Avaliação
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')