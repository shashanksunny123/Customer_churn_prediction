import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    stop = set(stopwords.words('english'))
    return " ".join([w for w in word_tokenize(text.lower())
                     if w.isalpha() and w not in stop])


df = pd.read_csv("data.csv")
df['text'] = df['text'].apply(preprocess)

df['label'] = LabelEncoder().fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# 🔹 BoW
bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.fit_transform(X_test)

model = MLPClassifier(max_iter=500)
model.fit(X_train_bow, y_train)

print("BoW Accuracy:",
      accuracy_score(y_test, model.predict(X_test_bow)))

# 🔹 TF-IDF
tfidf = TfidfVectorizer()
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)

model.fit(X_train_tf, y_train)

print("TF-IDF Accuracy:",
      accuracy_score(y_test, model.predict(X_test_tf)))