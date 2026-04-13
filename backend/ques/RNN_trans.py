import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🔹 Load dataset
df = pd.read_csv("data.csv")
eng, fra = df['eng'].tolist(), df['fra'].tolist()

# 🔹 Tokenization
tok_eng, tok_fra = Tokenizer(), Tokenizer()
tok_eng.fit_on_texts(eng)
tok_fra.fit_on_texts(fra)

X = pad_sequences(tok_eng.texts_to_sequences(eng), padding='post')
y = pad_sequences(tok_fra.texts_to_sequences(fra), padding='post')

# 🔹 Params
vocab_eng = len(tok_eng.word_index) + 1
vocab_fra = len(tok_fra.word_index) + 1
max_len_eng, max_len_fra = X.shape[1], y.shape[1]

# 🔹 Encoder
enc_in = Input(shape=(max_len_eng,))
enc_emb = Embedding(vocab_eng, 64)(enc_in)
_, h, c = LSTM(64, return_state=True)(enc_emb)

# 🔹 Decoder
dec_in = Input(shape=(max_len_fra,))
dec_emb = Embedding(vocab_fra, 64)(dec_in)
dec_out = LSTM(64, return_sequences=True)(dec_emb, initial_state=[h, c])
dec_out = Dense(vocab_fra, activation='softmax')(dec_out)

# 🔹 Model
model = Model([enc_in, dec_in], dec_out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# 🔹 Train
model.fit([X, y], y, epochs=200, verbose=0)

print("Training Done ✔️")