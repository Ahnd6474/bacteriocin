import numpy as np
import tensorflow as tf
import streamlit as st
import os
import pickle
from sklearn.metrics import accuracy_score

# 모델 로드
def load_models():
    model_path = 'model'
    with open(os.path.join(model_path, 'ensemble_model.pkl'), 'rb') as f:
        ml_model = pickle.load(f)

    mlp_model = tf.keras.models.load_model(os.path.join(model_path, 'mlp_model.h5'))
    cnn_model = tf.keras.models.load_model(os.path.join(model_path, 'cnn_model.h5'))
    dl_model_emb = tf.keras.models.load_model(os.path.join(model_path, 'dl_model_emb.h5'))

    return ml_model, mlp_model, cnn_model, dl_model_emb

# 원-핫 인코딩 함수
def one_hot_encode_sequence(sequence, max_length=300):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_dict = {aa: idx for idx, aa in enumerate(amino_acids)}
    encoded_seq = np.zeros((max_length, len(amino_acids)))

    for i, aa in enumerate(sequence):
        if i >= max_length:
            break
        if aa in aa_dict:
            encoded_seq[i, aa_dict[aa]] = 1.0

    return encoded_seq

# 모델 평가 함수
def evaluate_model(model, input_data, model_type='ml'):
    try:
        if model_type == 'ml':
            y_pred_prob = model.predict_proba(input_data.reshape(1, -1))[:, 1]
        else:
            y_pred_prob = model.predict(input_data).flatten()
        return y_pred_prob
    except Exception as e:
        st.error(f"Error evaluating {model_type} model: {e}")
        return np.array([])

# 모델의 정확도 로드
def load_model_accuracies():
    accuracies = {
        'ml': 0.981,   # XGBoost 정확도
        'mlp': 0.986,  # MLP 정확도
        'cnn': 0.985,  # CNN 정확도
        'dl': 0.984    # 임베딩 모델 정확도
    }
    return accuracies

# 가중치를 적용한 앙상블 평가 함수
def evaluate_ensemble(models, input_data, model_accuracies):
    preds = []
    weights = []
    for model, model_type, input_transform in models:
        try:
            transformed_input = input_transform(input_data)
            y_pred_prob = evaluate_model(model, transformed_input, model_type)
            if y_pred_prob.size > 0:
                preds.append(y_pred_prob * model_accuracies[model_type])
                weights.append(model_accuracies[model_type])
        except Exception as e:
            st.error(f"Error evaluating {model_type} model: {e}")

    if len(preds) == 0:
        st.error("No predictions could be made. Please check the input and try again.")
        return None

    preds = np.array(preds)
    weights = np.array(weights)
    y_pred_prob_final = np.average(preds, axis=0, weights=weights)

    return y_pred_prob_final

# Streamlit UI
st.title('Bacteriocin Amino Acid Sequence Classifier')
st.write('Enter an amino acid sequence to predict the probability it is a bacteriocin.')

sequence = st.text_area('Enter an amino acid sequence:')

if st.button('Classify'):
    if sequence:
        sequence = sequence.strip().upper()

        input_data = one_hot_encode_sequence(sequence)

        ml_model, mlp_model, cnn_model, dl_model_emb = load_models()
        model_accuracies = load_model_accuracies()

        def transform_ml(input_data):
            return input_data.flatten()[:300]

        def transform_mlp(input_data):
            return input_data.flatten()[:300].reshape(1, 300)

        def transform_cnn(input_data):
            return input_data.reshape(1, 300, 20, 1)

        def transform_dl(input_data):
            return input_data.flatten()[:100].reshape(1, 100)

        models = [
            (ml_model, 'ml', transform_ml),
            (mlp_model, 'mlp', transform_mlp),
            (cnn_model, 'cnn', transform_cnn),
            (dl_model_emb, 'dl', transform_dl)
        ]

        ensemble_prob = evaluate_ensemble(models, input_data, model_accuracies)

        if ensemble_prob is not None:
            st.write(f"Probability of being bacteriocin: {ensemble_prob[0] * 100:.2f}%")
    else:
        st.write('Please enter an amino acid sequence.')
