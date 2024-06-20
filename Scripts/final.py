import pickle
import tensorflow as tf
import os

# 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model')

# 모델 로드 함수
def load_models():
    ml_model = pickle.load(open(os.path.join(MODEL_PATH, 'ensemble_model.pkl'), 'rb'))
    mlp_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'mlp_model.h5'))
    cnn_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'cnn_model.h5'))
    dl_model_emb = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'dl_model_emb.h5'))
    return ml_model, mlp_model, cnn_model, dl_model_emb

# 모델 정확도 로드 함수 (이 함수는 실제 구현에 따라 달라질 수 있습니다)
def load_model_accuracies():
    return [0.98, 0.97, 0.96, 0.95]  # 예시로 모델 정확도를 반환

# 모델 저장 함수
def save_ensemble_model(models, model_accuracies, filename='final_model.pkl'):
    ensemble_data = {
        'models': models,
        'accuracies': model_accuracies
    }
    with open(filename, 'wb') as f:
        pickle.dump(ensemble_data, f)

# 입력 데이터 변환 함수
def transform_ml(input_data):
    return input_data.flatten()[:300]

def transform_mlp(input_data):
    return input_data.flatten()[:300].reshape(1, 300)

def transform_cnn(input_data):
    return input_data.reshape(1, 300, 20, 1)

def transform_dl(input_data):
    return input_data[:100].reshape(1, 100, 20)

# 메인 함수
def main():
    ml_model, mlp_model, cnn_model, dl_model_emb = load_models()
    model_accuracies = load_model_accuracies()

    models = [
        (ml_model, 'ml', transform_ml),
        (mlp_model, 'mlp', transform_mlp),
        (cnn_model, 'cnn', transform_cnn),
        (dl_model_emb, 'dl', transform_dl)
    ]

    save_ensemble_model(models, model_accuracies)

if __name__ == "__main__":
    main()
