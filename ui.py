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
            y_pred_prob = model.predict_proba(input_data)[:, 1]
        else:
            y_pred_prob = model.predict(input_data).flatten()
        return y_pred_prob
    except Exception as e:
        st.error(f"Error evaluating {model_type} model: {e}")
        return np.array([])

# 앙상블 평가 함수
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

# Streamlit UI
st.title('Bacteriocin Amino Acid Sequence Classifier')
st.write('Enter an amino acid sequence to predict the probability it is a bacteriocin.')

sequence = st.text_area('Enter an amino acid sequence:')

if st.button('Classify'):
    if sequence:
        sequence = sequence.strip().upper()

        input_data = np.array([one_hot_encode_sequence(sequence)])

        ml_model, mlp_model, cnn_model, dl_model_emb = load_models()

        models = [
            (ml_model, 'ml', None),
            (mlp_model, 'dl', None),
            (cnn_model, 'cnn', None),
            (dl_model_emb, 'dl', None)
        ]

        ensemble_prob = evaluate_ensemble(models, input_data)

        if ensemble_prob is not None:
            st.write(f"Probability of being bacteriocin: {ensemble_prob[0] * 100:.2f}%")
    else:
        st.write('Please enter an amino acid sequence.')
