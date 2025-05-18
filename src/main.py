import os
import sys
import pandas as pd
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from flask import Flask, send_from_directory, jsonify
from src.routes.prediction import create_prediction_route

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

APPLICANTS_PATH = "./data/applicants.json"


def load_json_to_df(file_path):
    # Carregar um JSON para um dataframe
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index.name = 'ID'
        df = df.reset_index()
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo {file_path} não encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {e}")
        return None


# Tratamento dos aplicantes a serem usados na prediction
applicants_df = load_json_to_df(APPLICANTS_PATH)
applicants_df = applicants_df.rename(columns={'ID': 'ID_APPLICANT_raw', 'infos_basicas': 'infos_basicas_applicant'})
if 'infos_basicas_applicant' in applicants_df.columns:
    applicants_df['ID_APPLICANT'] = applicants_df['infos_basicas_applicant'].apply(
        lambda x: x.get('codigo_profissional') if isinstance(x, dict) else None)
else:
    print("Coluna 'infos_basicas_applicant' não encontrada em applicants_df após carregamento.")
    if 'ID_APPLICANT_raw' in applicants_df.columns:
        applicants_df['ID_APPLICANT'] = applicants_df['ID_APPLICANT_raw']
    else:
        print("Não foi possível determinar ID_APPLICANT em applicants_df.")
applicants_df = applicants_df.drop(columns=["infos_basicas_applicant", "informacoes_pessoais",
                                            "cargo_atual", "cv_en", "ID_APPLICANT_raw"])
if 'informacoes_profissionais' in applicants_df.columns:
    try:
        info_prof_expanded = pd.json_normalize(applicants_df['informacoes_profissionais'].fillna({}))
        info_prof_expanded.columns = [f"app_prof_{col}" for col in info_prof_expanded.columns]
        applicants_df = pd.concat([applicants_df.drop(columns=['informacoes_profissionais']), info_prof_expanded], axis=1)
    except Exception as e:
        print(f"Erro ao normalizar 'informacoes_profissionais': {e}")

if 'formacao_e_idiomas' in applicants_df.columns:
    try:
        form_idiomas_expanded = pd.json_normalize(applicants_df['formacao_e_idiomas'].fillna({}))
        form_idiomas_expanded.columns = [f"app_form_{col}" for col in form_idiomas_expanded.columns]
        applicants_df = pd.concat([applicants_df.drop(columns=['formacao_e_idiomas']), form_idiomas_expanded], axis=1)
    except Exception as e:
        print(f"Erro ao normalizar 'formacao_e_idiomas': {e}")
print(applicants_df.columns)

applicants = applicants_df

# Registrar a rota de prediction
app = create_prediction_route(app)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return jsonify({"message": "API de recrutamento está no ar. Use o endpoint /predict para predições."}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

