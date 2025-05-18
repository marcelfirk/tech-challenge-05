import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Carregar o modelo
MODEL_PATH = "./modeltraining/model_rf.joblib"

try:
    model = joblib.load(MODEL_PATH)
    print(f"Modelo carregado de {MODEL_PATH}")
except FileNotFoundError:
    print(f"ERRO: Arquivo do modelo não encontrado em {MODEL_PATH}. A API de predição não funcionará.")
    model = None
except Exception as e:
    print(f"ERRO ao carregar o modelo de {MODEL_PATH}: {e}")
    model = None

# Lista de features esperadas pelo modelo
EXPECTED_FEATURES = [
    "cv_pt", # Text
    "vaga_principais_atividades", # Text
    "vaga_competencia_tecnicas_e_comportamentais", # Text
    "app_prof_conhecimentos_tecnicos", # Text
    "vaga_nivel profissional", # Categorical
    "vaga_nivel_academico", # Categorical
    "vaga_nivel_ingles", # Categorical
    "vaga_nivel_espanhol", # Categorical
    "vaga_local_trabalho", # Categorical
    "vaga_vaga_especifica_para_pcd", # Categorical
    "app_prof_nivel_profissional", # Categorical
    "app_form_nivel_academico", # Categorical
    "app_form_nivel_ingles", # Categorical
    "app_form_nivel_espanhol" # Categorical
]

TEXT_FEATURES_FOR_PREDICTION = [
    "cv_pt", 
    "vaga_principais_atividades", 
    "vaga_competencia_tecnicas_e_comportamentais", 
    "app_prof_conhecimentos_tecnicos"
]

CATEGORICAL_FEATURES_FOR_PREDICTION = [
    "vaga_nivel profissional", "vaga_nivel_academico", "vaga_nivel_ingles", "vaga_nivel_espanhol",
    "vaga_local_trabalho", "vaga_vaga_especifica_para_pcd",
    "app_prof_nivel_profissional", 
    "app_form_nivel_academico", "app_form_nivel_ingles", "app_form_nivel_espanhol"
]


def create_prediction_route(app):
    from ..main import applicants

    @app.route("/predict", methods=["POST"])
    def predict():
        if model is None:
            return jsonify({"error": "Modelo não carregado. Predição indisponível."}), 500

        try:
            # Receber os dados da vaga
            vaga_data = request.get_json(silent=True)
            if vaga_data is None:
                return jsonify({"error": "Payload JSON inválido ou Content-Type incorreto. Use application/json."}), 400

            # Verificar se todos os campos da vaga estão presentes
            missing_fields = [feature for feature in EXPECTED_FEATURES if feature.startswith("vaga_") and feature not in vaga_data]
            if missing_fields:
                return jsonify({"error": f"Campos ausentes no payload da vaga: {missing_fields}"}), 400

            # Aplicar pré-processamento nos campos da vaga
            for col in TEXT_FEATURES_FOR_PREDICTION:
                if col in vaga_data:
                    vaga_data[col] = vaga_data.get(col, "")

            for col in CATEGORICAL_FEATURES_FOR_PREDICTION:
                if col in vaga_data:
                    vaga_data[col] = vaga_data.get(col, "Desconhecido")

            # Lista para armazenar os registros combinados
            input_data = []

            # Iterar sobre cada linha de "applicants"
            for _, applicant_row in applicants.iterrows():
                # Combinar os dados da vaga com os dados do candidato
                combined_record = vaga_data.copy()

                # Preencher os dados do candidato
                for feature in applicants.columns:
                    if feature in EXPECTED_FEATURES:
                        combined_record[feature] = applicant_row[feature]

                # Adicionar ao conjunto de entrada
                input_data.append(combined_record)

            # Criar DataFrame de entrada
            input_df = pd.DataFrame(input_data, columns=EXPECTED_FEATURES)

            for col in TEXT_FEATURES_FOR_PREDICTION:
                input_df[col] = input_df[col].fillna("")

            for col in CATEGORICAL_FEATURES_FOR_PREDICTION:
                input_df[col] = input_df[col].fillna("Desconhecido").astype(str)

            # Realizar a predição
            predictions = model.predict(input_df)
            probabilities = model.predict_proba(input_df)

            # Formatar a resposta
            results = []
            for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
                results.append({
                    "index": i,
                    "applicant_index": applicants.loc[i,'ID_APPLICANT'],  # Posição do candidato no DataFrame
                    "prediction": int(pred),
                    "probability_no_match": float(proba[0]),
                    "probability_match": float(proba[1])
                })

            # Pegar os 5 maiores matches por probabilidade de Target = 1
            top_5_matches = sorted(results, key=lambda x: x["probability_match"], reverse=True)[:5]

            return jsonify(top_5_matches), 200

        except Exception as e:
            print(f"Erro durante a predição: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Erro interno do servidor durante a predição: {str(e)}"}), 500

    return app

