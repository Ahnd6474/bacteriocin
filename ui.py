import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'Data', 'processed')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model')

# 데이터 로드 함수
@st.cache_data
def load_data():
    x_test_path = os.path.join(DATA_PATH, 'X_test.pkl')
    y_test_path = os.path.join(DATA_PATH, 'y_test.pkl')

    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        st.error(f"File not found: {x_test_path} or {y_test_path}")
        st.stop()

    with open(x_test_path, 'rb') as f:
        X_test = pickle.load(f)
    with open(y_test_path, 'rb') as f:
        y_test = pickle.load(f)

    # y_test를 numpy 배열로 변환
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    return X_test, y_test

# 모델 로드 함수
@st.cache_resource
def load_models():
    ensemble_model_path = os.path.join(MODEL_PATH, 'ensemble_model.pkl')
    mlp_model_path = os.path.join(MODEL_PATH, 'mlp_model.h5')
    cnn_model_path = os.path.join(MODEL_PATH, 'cnn_model.h5')
    dl_model_emb_path = os.path.join(MODEL_PATH, 'dl_model_emb.h5')

    if not os.path.exists(ensemble_model_path) or not os.path.exists(mlp_model_path) or not os.path.exists(cnn_model_path) or not os.path.exists(dl_model_emb_path):
        st.error(f"Model file not found in path: {MODEL_PATH}")
        st.stop()

    with open(ensemble_model_path, 'rb') as f:
        ml_model = pickle.load(f)
    mlp_model = tf.keras.models.load_model(mlp_model_path)
    cnn_model = tf.keras.models.load_model(cnn_model_path)
    dl_model_emb = tf.keras.models.load_model(dl_model_emb_path)
    return ml_model, mlp_model, cnn_model, dl_model_emb

# 모델 평가 함수
def evaluate_model(model, input_data, model_type='ml'):
    try:
        if model_type == 'ml':
            y_pred_prob = model.predict_proba(input_data)[:, 1]
        else:
            y_pred_prob = model.predict(input_data).flatten()
        return y_pred_prob
    except Exception as e:
        st.error(f"Error evaluating {model_type} model: {e}")
        return np.array([])

# 가중치 투표 방식 평가 함수
def evaluate_ensemble(models, input_data):
    preds = []
    for model, model_type, _ in models:
        try:
            if model_type == 'cnn':
                cnn_input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)
                y_pred_prob = evaluate_model(model, cnn_input_data, model_type)
            elif model_type == 'dl' and model.input_shape[-1] == 100:
                dl_input_data = input_data[:, :100]
                y_pred_prob = evaluate_model(model, dl_input_data, model_type)
            elif model_type == 'dl' and model.input_shape[-1] == 300:
                dl_input_data = input_data
                y_pred_prob = evaluate_model(model, dl_input_data, model_type)
            else:
                y_pred_prob = evaluate_model(model, input_data, model_type)
            if y_pred_prob.size > 0:
                preds.append(y_pred_prob)
        except Exception as e:
            st.error(f"Error evaluating {model_type} model: {e}")

    if len(preds) == 0:
        st.error("No predictions could be made. Please check the input and try again.")
        return None

    preds = np.array(preds)
    y_pred_prob_final = np.mean(preds, axis=0)

    return y_pred_prob_final

# 스트림릿 UI 설정
st.title('Bacteriocin Amino Acid Sequence Classifier')
st.write('Enter amino acid sequences to predict the probability they are bacteriocins.')

# 아미노산 서열 입력
sequences = st.text_area('Enter amino acid sequences (one per line):')

if st.button('Classify'):
    if sequences:
        sequences = sequences.strip().split('\n')

        # 데이터 로드
        X_test, y_test = load_data()

        # 모델 로드
        ml_model, mlp_model, cnn_model, dl_model_emb = load_models()

        # 모델 및 가중치 설정
        models = [
            (ml_model, 'ml', None),
            (mlp_model, 'dl', None),
            (cnn_model, 'cnn', None),
            (dl_model_emb, 'dl', None)
        ]

        # 모델 평가 및 집계
        ensemble_probs = evaluate_ensemble(models, X_test)

        # 결과 출력
        if ensemble_probs is not None:
            for i, prob in enumerate(ensemble_probs):
                st.write(f"Sequence {i+1}: Probability of being bacteriocin: {prob:.4f}")
    else:
        st.write('Please enter at least one amino acid sequence.')
