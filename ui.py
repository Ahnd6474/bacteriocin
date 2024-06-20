import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
    if model_type == 'ml':
        y_pred = model.predict(input_data)
        y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    else:
        y_pred = (model.predict(input_data) > 0.5).astype("int32")
        y_pred = y_pred.flatten()
    return y_pred

# 가중치 투표 방식 평가 함수
def evaluate_ensemble(models, input_data, y_test):
    preds = []
    weights = []
    for model, model_type in models:
        if model_type == 'cnn':
            cnn_input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)
            y_pred = evaluate_model(model, cnn_input_data, model_type)
        elif model_type == 'dl' and model.input_shape[-1] == 100:
            dl_input_data = input_data[:, :100]
            y_pred = evaluate_model(model, dl_input_data, model_type)
        elif model_type == 'dl' and model.input_shape[-1] == 300:
            dl_input_data = input_data
            y_pred = evaluate_model(model, dl_input_data, model_type)
        else:
            y_pred = evaluate_model(model, input_data, model_type)
        preds.append(y_pred)

    preds = np.array(preds)
    y_pred_final = np.zeros(preds.shape[1])
    for i in range(len(models)):
        y_pred_final += preds[i]

    y_pred_final = (y_pred_final / len(models)).round().astype(int)

    # y_test의 형식을 일관되게 맞춤
    y_test = y_test.astype(int)

    accuracy = accuracy_score(y_test, y_pred_final)
    cm = confusion_matrix(y_test, y_pred_final)
    cr = classification_report(y_test, y_pred_final)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)

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

        # 모델 정확도 계산
        ml_accuracy = accuracy_score(y_test, evaluate_model(ml_model, X_test, 'ml'))
        mlp_accuracy = accuracy_score(y_test, evaluate_model(mlp_model, X_test, 'dl'))
        cnn_accuracy = accuracy_score(y_test, evaluate_model(cnn_model, X_test.reshape(X_test.shape[0], X_test.shape[1], 1), 'cnn'))
        dl_accuracy = accuracy_score(y_test, evaluate_model(dl_model_emb, X_test, 'dl'))

        # 모델 및 가중치 설정
        models = [
            (ml_model, 'ml', ml_accuracy),
            (mlp_model, 'dl', mlp_accuracy),
            (cnn_model, 'cnn', cnn_accuracy),
            (dl_model_emb, 'dl', dl_accuracy)
        ]

        # 모델 평가 및 집계
        ensemble_accuracy = evaluate_ensemble(models, X_test, y_test)

        # 결과 출력
        st.write('Ensemble Model Accuracy:')
        st.write(f'{ensemble_accuracy}')
    else:
        st.write('Please enter at least one amino acid sequence.')
