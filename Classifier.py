#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Code by Prithviraj | GitHub - PR-ODINSON

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
import nltk
# Code by Prithviraj | GitHub - PR-ODINSON

from nltk.corpus import stopwords
import re
import pickle
from imblearn.over_sampling import SMOTE

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# Code by Prithviraj | GitHub - PR-ODINSON

# Load dataset
data = pd.read_csv('combined_data.csv')  

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Code by Prithviraj | GitHub - PR-ODINSON
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing
data['text'] = data['text'].apply(preprocess_text)
data = data[data['text'] != ""]
data = data.drop_duplicates(subset=['text', 'label'])
# Code by Prithviraj | GitHub - PR-ODINSON

# Features and labels
X = data['text'].values
y = data['label'].values

# Tokenization
max_words = 25000  # Increased vocabulary size
max_len = 120  # Increased sequence length
# Code by Prithviraj | GitHub - PR-ODINSON
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

X_pad = pad_sequences(X_seq, maxlen=max_len)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_pad_resampled, y_resampled = smote.fit_resample(X_pad, y)
# Code by Prithviraj | GitHub - PR-ODINSON

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pad_resampled, y_resampled, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),  # Increased embedding size
    SpatialDropout1D(0.3),  # Helps with overfitting
    # Code by Prithviraj | GitHub - PR-ODINSON
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),  # Added dropout
    Dense(64, activation='relu'),
    Dropout(0.5),  # Increased dropout
    Dense(1, activation='sigmoid')
])

# Compile the model with reduced learning rate
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Code by Prithviraj | GitHub - PR-ODINSON

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
# Code by Prithviraj | GitHub - PR-ODINSON

# Save the model and tokenizer
model.save('spam_classifier_lstm_tuned.h5')
with open('tokenizer_tuned.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Test on a sample email
sample_email = "Urgent: Your account has been compromised. Click here to secure it!"
sample_seq = tokenizer.texts_to_sequences([preprocess_text(sample_email)])
sample_pad = pad_sequences(sample_seq, maxlen=max_len)
sample_pred = model.predict(sample_pad)[0][0]
print("Sample Prediction:", "Spam" if sample_pred >= 0.5 else "Ham")
# Code by Prithviraj | GitHub - PR-ODINSON

