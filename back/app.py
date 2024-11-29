from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
from flask_cors import CORS

# Inicializa a aplicação Flask
app = Flask(__name__, static_folder="../data_science_visualization/FRONTEND", static_url_path="", template_folder="../data_science_visualization/FRONTEND")

# Habilita o CORS para todas as rotas
CORS(app)

# Carrega o modelo treinado e o scaler
model = joblib.load("./gradient_boosting_model_v1.joblib")
scaler = joblib.load("./scaler.joblib")  # Carrega o scaler (se usado)

# Rota para a página inicial do front-end
@app.route('/')
def index():
    return send_from_directory('../data_science_visualization/FRONTEND', 'index.html')

# Rota para a API de predição
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Dados recebidos do front-end
        data = request.get_json()

        # Definir as features esperadas
        EXPECTED_FEATURES = ["latitude", "longitude", "numero_dias_sem_chuva", "precipitacao"]

        # Validar se todas as features estão presentes
        missing_features = [feature for feature in EXPECTED_FEATURES if feature not in data]
        if missing_features:
            return jsonify({"error": f"Faltam variáveis: {', '.join(missing_features)}"}), 400

        # Criar o DataFrame e garantir que as colunas estão na ordem correta
        df = pd.DataFrame([data])[EXPECTED_FEATURES]

        # Aplicar o scaler, se necessário
        df_scaled = scaler.transform(df)

        # Predição
        prediction = model.predict(df_scaled)

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
