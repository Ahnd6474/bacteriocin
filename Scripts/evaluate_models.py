import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
import os

# 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'Data', 'processed')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model')

# 데이터 로드
with open(os.path.join(DATA_PATH, 'X_test.pkl'), 'rb') as f:
    X_test = pickle.load(f)
with open(os.path.join(DATA_PATH, 'y_test.pkl'), 'rb') as f:
    y_test = pickle.load(f)

# 모델 로드
with open(os.path.join(MODEL_PATH, 'ensemble_model.pkl'), 'rb') as f:
    ml_model = pickle.load(f)

mlp_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'mlp_model.h5'))
cnn_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'cnn_model.h5'))
dl_model_emb = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'dl_model_emb.h5'))

# y_test의 형식 변환 (필요할 경우)
if isinstance(y_test, pd.Series):
    y_test = y_test.to_numpy().astype(int)

# 모델 평가 함수
def evaluate_model(model, X_test, y_test, model_type='ml'):
    if model_type == 'ml':
        y_pred = model.predict(X_test)
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = (y_pred > 0.5).astype("int32")
    else:
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        y_pred = y_pred.flatten()

    y_test = y_test.flatten() if hasattr(y_test, 'flatten') else y_test

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)

    return accuracy

# 머신러닝 모델 평가
ml_accuracy = evaluate_model(ml_model, X_test, y_test, model_type='ml')

# MLP 모델 평가
mlp_accuracy = evaluate_model(mlp_model, X_test, y_test, model_type='dl')

# CNN 모델 평가
cnn_X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)  # CNN input shape 맞추기
cnn_accuracy = evaluate_model(cnn_model, cnn_X_test, y_test, model_type='dl')

# 임베딩 모델 평가
# 입력 크기 맞추기
dl_input_shape = dl_model_emb.input_shape[1]
if X_test.shape[1] != dl_input_shape:
    # X_test의 두 번째 차원이 dl_input_shape와 다를 경우 맞춤
    X_test_emb = np.zeros((X_test.shape[0], dl_input_shape))
    X_test_emb[:, :min(dl_input_shape, X_test.shape[1])] = X_test[:, :min(dl_input_shape, X_test.shape[1])]
else:
    X_test_emb = X_test

dl_model_emb_accuracy = evaluate_model(dl_model_emb, X_test_emb, y_test, model_type='dl')

print(f"Machine Learning Model Accuracy: {ml_accuracy}")
print(f"MLP Model Accuracy: {mlp_accuracy}")
print(f"CNN Model Accuracy: {cnn_accuracy}")
print(f"Deep Learning Model with Embeddings Accuracy: {dl_model_emb_accuracy}")
