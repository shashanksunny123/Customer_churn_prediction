import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🔹 Load corpus
text = open("corpus.txt").read().lower()

# 🔹 Tokenize
tok = Tokenizer()
tok.fit_on_texts([text])
seq = tok.texts_to_sequences([text])[0]

# 🔹 Create sequences
X, y = [], []
for i in range(1, len(seq)):
    X.append(seq[:i])
    y.append(seq[i])

X = pad_sequences(X)
y = np.array(y)

vocab = len(tok.word_index) + 1

# 🔹 Model
model = Sequential([
    Embedding(vocab, 50, input_length=X.shape[1]),
    LSTM(100),
    Dense(vocab, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=200, verbose=0)

# 🔹 Generate text
def generate(seed, n=5):
    for _ in range(n):
        seq = tok.texts_to_sequences([seed])[0]
        seq = pad_sequences([seq], maxlen=X.shape[1])
        pred = model.predict(seq, verbose=0)
        word = next((w for w,i in tok.word_index.items() if i==np.argmax(pred)), "")
        seed += " " + word
    return seed

print(generate("machine", 5))