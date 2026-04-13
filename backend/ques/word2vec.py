from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd

sentences = [
    "data science is an interdisciplinary field",
    "machine learning models improve with data",
    "deep learning uses neural networks",
    "natural language processing deals with text data",
    "artificial intelligence is transforming industries",
    "python is widely used in data analysis",
    "big data requires efficient processing techniques",
    "algorithms are essential for problem solving",
    "statistics helps in understanding data patterns",
    "cloud computing enables scalable applications"
]

words=[word_tokenize(text.lower()) for text in sentences]
model=Word2Vec(words, vector_size=100, window=5, min_count=1, workers=4, epochs=100)

print("Words similar to machine")
print(model.wv.most_similar('machine'))

print("\n vector for data")
print(model.wv['data'])