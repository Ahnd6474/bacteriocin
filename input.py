import streamlit as st
import pickle
import numpy as np
import tensorflow as tf

# 모델 로드 함수
def load_final_model(filename='model/final_model.pkl'):
    with open(filename, 'rb') as f:
        ensemble_data = pickle.load(f)
    return ensemble_data['models'], ensemble_data['accuracies']

# 입력 데이터 변환 함수
def one_hot_encode_sequence(sequence, max_length=300):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    encoding = np.zeros((max_length, len(amino_acids)))
    for i, aa in enumerate(sequence[:max_length]):
        if aa in amino_acids:
            encoding[i, amino_acids.index(aa)] = 1
    return encoding

# 모델 평가 함수
def evaluate_model(model, input_data, model_type):
    try:
        if model_type == 'ml':
            y_pred_prob = model.predict_proba(input_data)[:, 1]
        else:
            y_pred_prob = model.predict(input_data)
            if y_pred_prob.ndim > 1:
                y_pred_prob = y_pred_prob[:, 0]
        return y_pred_prob
    except Exception as e:
        print(f"Error evaluating {model_type} model: {str(e)}")
        return None

# 앙상블 평가 함수
def evaluate_ensemble(models, input_data, model_accuracies):
    y_pred_probs = []
    weights = []

    for model, model_type, transform_func in models:
        transformed_input = transform_func(input_data)
        y_pred_prob = evaluate_model(model, transformed_input, model_type)
        if y_pred_prob is not None:
            y_pred_probs.append(y_pred_prob)
            weights.append(model_accuracies[models.index((model, model_type, transform_func))])

    if not y_pred_probs:
        return None

    y_pred_probs = np.array(y_pred_probs)
    weights = np.array(weights) / np.sum(weights)
    y_pred_prob_final = np.average(y_pred_probs, axis=0, weights=weights)

    return y_pred_prob_final

# Streamlit UI 구성
st.title('Bacteriocin Prediction')

sequence = st.text_area("Enter an amino acid sequence:")

if st.button('Predict'):
    if sequence:
        input_data = one_hot_encode_sequence(sequence)

        models, model_accuracies = load_final_model()

        y_pred_prob_final = evaluate_ensemble(models, input_data, model_accuracies)

        if y_pred_prob_final is not None:
            st.write(f"Probability of being bacteriocin: {y_pred_prob_final[0] * 100:.2f}%")
        else:
            st.write("No predictions could be made. Please check the input and try again.")
    else:
        st.write("Please enter an amino acid sequence.")
