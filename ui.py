import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score

# 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'Data', 'processed')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model')

# 데이터 로드 함수
@st.cache_data
def load_data():
    with open(os.path.join(DATA_PATH, 'X_test.pkl'), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(DATA_PATH, 'y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    return X_test, y_test

# 모델 로드 함수
@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_PATH, 'ensemble_model.pkl'), 'rb') as f:
        ml_model = pickle.load(f)
    mlp_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'mlp_model.h5'))
    cnn_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'cnn_model.h5'))
    dl_model_emb = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'dl_model_emb.h5'))
    return ml_model, mlp_model, cnn_model, dl_model_emb

# 모델 평가 함수
def evaluate_model(model, input_data, model_type='ml'):
    if model_type == 'ml':
        y_pred = model.predict(input_data)
        y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    else:
        y_pred = (model.predict(input_data) > 0.5).astype("int32")
        y_pred = y_pred.flatten()
    return y_pred

# 모델 평가 함수 (집계 결과 포함)
def evaluate_ensemble(models, input_data, y_test):
    preds = []
    for model, model_type in models:
        y_pred = evaluate_model(model, input_data, model_type)
        preds.append(y_pred)

    # Majority voting
    preds = np.array(preds)
    y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# 스트림릿 UI 설정
st.title('Bacteriocin Amino Acid Sequence Classifier')
st.write('Enter amino acid sequences to predict whether they are bacteriocins.')

# 아미노산 서열 입력
sequences = st.text_area('Enter amino acid sequences (one per line):')

if st.button('Classify'):
    if sequences:
        sequences = sequences.strip().split('\n')

        # 데이터 로드
        X_test, y_test = load_data()

        # 모델 로드
        ml_model, mlp_model, cnn_model, dl_model_emb = load_models()

        # 머신러닝 모델 평가
        ml_accuracy = evaluate_model(ml_model, X_test, model_type='ml')

        # MLP 모델 평가
        mlp_accuracy = evaluate_model(mlp_model, X_test, model_type='dl')

        # CNN 모델 평가
        cnn_X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        cnn_accuracy = evaluate_model(cnn_model, cnn_X_test, model_type='dl')

        # 임베딩 모델 평가
        dl_input_data = X_test.reshape(X_test.shape[0], 100, 3)
        dl_accuracy = evaluate_model(dl_model_emb, dl_input_data, model_type='dl')

        # 모델 로드 및 평가 (집계)
        models = [(ml_model, 'ml'), (mlp_model, 'dl'), (cnn_model, 'dl'), (dl_model_emb, 'dl')]
        ensemble_accuracy = evaluate_ensemble(models, X_test, y_test)

        # 결과 출력
        st.write('Ensemble Model Accuracy:')
        st.write(f'{ensemble_accuracy}')
    else:
        st.write('Please enter at least one amino acid sequence.')
