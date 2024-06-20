# Import necessary libraries
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
import pickle
from gensim.models import Word2Vec
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Ensure directories exist
os.makedirs(r'C:\Users\User\PycharmProjects\Bacteriocin2\model', exist_ok=True)
os.makedirs(r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\processed', exist_ok=True)

# Load data function
def load_data():
    data = pd.read_csv(r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\processed\augmented_data.csv')
    return data

# Train Word2Vec model
def train_word2vec(sequences, vector_size=100, window=5, min_count=1, workers=4):
    sequences_split = [list(sequence) for sequence in sequences]
    model = Word2Vec(sentences=sequences_split, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

# Get embeddings for sequences
def get_sequence_embeddings(sequences, model, vector_size):
    embeddings = []
    for sequence in sequences:
        embedding = np.mean([model.wv[char] for char in sequence if char in model.wv], axis=0)
        embeddings.append(embedding)
    return np.array(embeddings)

# Load and preprocess data
data = load_data()

# Ensure labels are discrete classes
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])

# Experiment with different n-gram ranges
ngram_ranges = [(2, 2), (3, 3), (4, 4)]
best_accuracy = 0
best_ngram_range = None
best_vectorizer = None

for ngram_range in ngram_ranges:
    vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range, max_features=10000)
    X = vectorizer.fit_transform(data['Sequence'])
    y = data['Label']
    input_dim = X.shape[1]

    # Split data and apply SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train and evaluate a simple model
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_res, y_train_res)
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_ngram_range = ngram_range
        best_vectorizer = vectorizer

print(f'Best n-gram range: {best_ngram_range} with accuracy: {best_accuracy}')

# Save the best vectorizer
with open(r'C:\Users\User\PycharmProjects\Bacteriocin2\model\vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(best_vectorizer, vec_file)

# Use the best vectorizer for final training
X = best_vectorizer.fit_transform(data['Sequence'])
y = data['Label']
input_dim = X.shape[1]

# Reduce dimensionality with TruncatedSVD
svd = TruncatedSVD(n_components=300)
X = svd.fit_transform(X)

# Split data and apply SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Word2Vec Embedding
sequences = data['Sequence'].tolist()
w2v_model = train_word2vec(sequences)
embeddings = get_sequence_embeddings(sequences, w2v_model, vector_size=100)

# Split data and apply SMOTE for embeddings
X_train_emb, X_test_emb, y_train_emb, y_test_emb = train_test_split(embeddings, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_res_emb, y_train_res_emb = smote.fit_resample(X_train_emb, y_train_emb)

# Save the trained Word2Vec model
w2v_model.save(r"C:\Users\User\PycharmProjects\Bacteriocin2\model\w2v_model.bin")

# Save the processed data
with open(r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\processed\X_train_res.pkl', 'wb') as f:
    pickle.dump(X_train_res, f)
with open(r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\processed\y_train_res.pkl', 'wb') as f:
    pickle.dump(y_train_res, f)
with open(r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\processed\X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open(r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\processed\y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
with open(r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\processed\X_train_res_emb.pkl', 'wb') as f:
    pickle.dump(X_train_res_emb, f)
with open(r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\processed\y_train_res_emb.pkl', 'wb') as f:
    pickle.dump(y_train_res_emb, f)
with open(r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\processed\X_test_emb.pkl', 'wb') as f:
    pickle.dump(X_test_emb, f)
with open(r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\processed\y_test_emb.pkl', 'wb') as f:
    pickle.dump(y_test_emb, f)
