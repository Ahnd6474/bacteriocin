import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

# Define paths
MODEL_PATH = r'C:\Users\User\PycharmProjects\Bacteriocin2\model'
DATA_PATH = r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\processed'

# Ensure the model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

# Hyperparameter tuning for XGBoost
def hyperparameter_tuning(X, y):
    print("Starting hyperparameter tuning for XGBoost...")
    param_distributions = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'learning_rate': [0.01, 0.05]
    }
    xgb = XGBClassifier(random_state=42)
    random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_distributions, n_iter=5, cv=3,
                                       verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X, y)
    print("Completed hyperparameter tuning for XGBoost.")
    return random_search.best_estimator_

# Train and save machine learning models with cross-validation
def train_and_save_ml_models(X, y):
    print("Starting training for machine learning models...")
    rf = RandomForestClassifier(random_state=42)
    xgb = hyperparameter_tuning(X, y)
    lgbm = LGBMClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(random_state=42, max_iter=1000)

    models = [
        ('Random Forest', rf),
        ('XGBoost', xgb),
        ('LightGBM', lgbm),
        ('Gradient Boosting', gb),
        ('Logistic Regression', lr)
    ]

    ensemble = VotingClassifier(estimators=models, voting='soft')

    # Perform cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validation scores for ensemble model: {cv_scores}")
    print(f"Mean CV score for ensemble model: {cv_scores.mean()}")

    # Fit the model on the entire dataset
    ensemble.fit(X, y)
    print("Completed training for machine learning models.")

    with open(f'{MODEL_PATH}/ensemble_model.pkl', 'wb') as model_file:
        pickle.dump(ensemble, model_file)
    print("Ensemble model saved.")
    return ensemble

# Create and train deep learning model (MLP)
def create_mlp_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_and_save_mlp_model(X, y, input_dim):
    print("Starting training for MLP model...")
    model = create_mlp_model(input_dim)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X, y, epochs=50, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])
    print("Completed training for MLP model.")
    model.save(f'{MODEL_PATH}/mlp_model.h5')
    print("MLP model saved.")
    return model

# Create and train CNN model
def create_cnn_model(input_dim):
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(input_dim, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_and_save_cnn_model(X, y, input_dim):
    print("Starting training for CNN model...")
    model = create_cnn_model(input_dim)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X, y, epochs=50, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])
    print("Completed training for CNN model.")
    model.save(f'{MODEL_PATH}/cnn_model.h5')
    print("CNN model saved.")
    return model

# Load processed data
print("Loading processed data...")
with open(f'{DATA_PATH}/X_train_res.pkl', 'rb') as f:
    X_train_res = pickle.load(f)
with open(f'{DATA_PATH}/y_train_res.pkl', 'rb') as f:
    y_train_res = pickle.load(f)
with open(f'{DATA_PATH}/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open(f'{DATA_PATH}/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
with open(f'{DATA_PATH}/X_train_res_emb.pkl', 'rb') as f:
    X_train_res_emb = pickle.load(f)
with open(f'{DATA_PATH}/y_train_res_emb.pkl', 'rb') as f:
    y_train_res_emb = pickle.load(f)
with open(f'{DATA_PATH}/X_test_emb.pkl', 'rb') as f:
    X_test_emb = pickle.load(f)
with open(f'{DATA_PATH}/y_test_emb.pkl', 'rb') as f:
    y_test_emb = pickle.load(f)
print("Data loaded.")

# Ensure labels are correctly encoded and converted to writable arrays
le = LabelEncoder()

# Make a writable copy of the data
y_train_res = np.copy(y_train_res)
y_test = np.copy(y_test)
y_train_res_emb = np.copy(y_train_res_emb)
y_test_emb = np.copy(y_test_emb)

# Transform labels
y_train_res = np.array(le.fit_transform(y_train_res), dtype=np.int32)
y_test = np.array(le.transform(y_test), dtype=np.int32)
y_train_res_emb = np.array(le.fit_transform(y_train_res_emb), dtype=np.int32)
y_test_emb = np.array(le.transform(y_test_emb), dtype=np.int32)

# Train and save models
ml_model = train_and_save_ml_models(X_train_res, y_train_res)
mlp_model = train_and_save_mlp_model(X_train_res, y_train_res, X_train_res.shape[1])

# Prepare data for CNN model
cnn_X_train_res = X_train_res.reshape((X_train_res.shape[0], X_train_res.shape[1], 1))
cnn_X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

cnn_model = train_and_save_cnn_model(cnn_X_train_res, y_train_res, X_train_res.shape[1])

# Train and save deep learning model with embeddings
def train_and_save_dl_model_with_embeddings(X_train_res, y_train_res, input_dim):
    print("Starting training for deep learning model with embeddings...")
    model = create_mlp_model(input_dim)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train_res, y_train_res, epochs=50, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])
    model.save(f'{MODEL_PATH}/dl_model_emb.h5')
    print("Deep learning model with embeddings saved.")
    return model

dl_model_emb = train_and_save_dl_model_with_embeddings(X_train_res_emb, y_train_res_emb, X_train_res_emb.shape[1])
