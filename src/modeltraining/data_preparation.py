import json
import pandas as pd

def load_json_to_df(file_path):
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

def prepare_data():
    print("Iniciando o carregamento dos dados...")
    vagas_df = load_json_to_df('../data/vagas.json')
    applicants_df = load_json_to_df('../data/applicants.json')

    try:
        with open('../data/prospects.json', 'r', encoding='utf-8') as f:
            prospects_data = json.load(f)
    except FileNotFoundError:
        print("Erro: Arquivo ./data/prospects.json não encontrado.")
        return
    except Exception as e:
        print(f"Erro ao carregar ./data/prospects.json: {e}")
        return

    if vagas_df is None or applicants_df is None or prospects_data is None:
        print("Falha ao carregar um ou mais arquivos de dados. Abortando preparação.")
        return

    print("Arquivos JSON carregados.")

    # Achatar dados dos prospects
    prospects_list = []
    for vaga_id, data in prospects_data.items():
        for prospect_info in data.get('prospects', []):
            prospects_list.append({
                'ID_VAGA': vaga_id,
                'ID_APPLICANT': prospect_info.get('codigo'),
                'situacao_candidado': prospect_info.get('situacao_candidado'),
                'titulo_vaga_prospect': data.get('titulo') # título da vaga no prospects.json
            })
    prospects_flat_df = pd.DataFrame(prospects_list)
    print(f"Prospects.json achatado. {len(prospects_flat_df)} registros de prospects.")

    # Renomear colunas para merge
    vagas_df = vagas_df.rename(columns={'ID': 'ID_VAGA'})
    applicants_df = applicants_df.rename(columns={'ID': 'ID_APPLICANT_raw', 'infos_basicas': 'infos_basicas_applicant'})
    if 'infos_basicas_applicant' in applicants_df.columns:
        applicants_df['ID_APPLICANT'] = applicants_df['infos_basicas_applicant'].apply(lambda x: x.get('codigo_profissional') if isinstance(x, dict) else None)
    else:
        print("Coluna 'infos_basicas_applicant' não encontrada em applicants_df após carregamento.")
        if 'ID_APPLICANT_raw' in applicants_df.columns:
             applicants_df['ID_APPLICANT'] = applicants_df['ID_APPLICANT_raw']
        else:
            print("Não foi possível determinar ID_APPLICANT em applicants_df.")
            return

    print("Colunas renomeadas para merge.")
    print(f"Colunas em vagas_df: {vagas_df.columns.tolist()}")
    print(f"Colunas em applicants_df: {applicants_df.columns.tolist()}")
    print(f"Colunas em prospects_flat_df: {prospects_flat_df.columns.tolist()}")
    print("Iniciando merge dos DataFrames...")
    # Merge prospects com vagas
    merged_df = pd.merge(prospects_flat_df, vagas_df, on='ID_VAGA', how='left')
    print(f"Tamanho após merge com vagas: {len(merged_df)} linhas.")
    # Merge com applicants
    merged_df = pd.merge(merged_df, applicants_df, on='ID_APPLICANT', how='left')
    print(f"Tamanho após merge com applicants: {len(merged_df)} linhas.")

    # Definir target
    positive_status = ['Contratado pela Decision', 'Aprovado', 'Proposta Aceita', 'Contratado como Hunting', 'Documentação PJ', 'Documentação Cooperado', 'Documentação CLT']
    negative_status = ['Não Aprovado pelo Requisitante', 'Não Aprovado pelo Cliente', 'Não Aprovado pelo RH', 'Recusado', 'Desistiu', 'Desistiu da Contratação', 'Sem interesse nesta vaga']

    merged_df['target'] = merged_df['situacao_candidado'].apply(
        lambda x: 1 if x in positive_status else (0 if x in negative_status else pd.NA)
    )
    
    # Remover linhas onde o target for nulo
    initial_rows = len(merged_df)
    merged_df.dropna(subset=['target'], inplace=True)
    merged_df['target'] = merged_df['target'].astype(int)
    rows_after_target_definition = len(merged_df)
    print(f"{initial_rows - rows_after_target_definition} linhas removidas devido a status de candidato não mapeados para target binário.")
    print(f"Dataset final para modelagem tem {rows_after_target_definition} linhas.")
    print(f"Distribuição da variável alvo:\n{merged_df['target'].value_counts(normalize=True)}")

    # Selecionar colunas relevantes para evitar sobrecarga de memória e focar no essencial
    cols_vagas = ['ID_VAGA', 'perfil_vaga']
    cols_applicants = ['ID_APPLICANT', 'informacoes_profissionais', 'formacao_e_idiomas', 'cv_pt']
    
    # Expandir dicts em colunas separadas
    if 'perfil_vaga' in merged_df.columns:
        try:
            perfil_vaga_expanded = pd.json_normalize(merged_df['perfil_vaga'].fillna({}))
            perfil_vaga_expanded.columns = [f"vaga_{col}" for col in perfil_vaga_expanded.columns]
            merged_df = pd.concat([merged_df.drop(columns=['perfil_vaga']), perfil_vaga_expanded], axis=1)
        except Exception as e:
            print(f"Erro ao normalizar 'perfil_vaga': {e}")

    if 'informacoes_profissionais' in merged_df.columns:
        try:
            info_prof_expanded = pd.json_normalize(merged_df['informacoes_profissionais'].fillna({}))
            info_prof_expanded.columns = [f"app_prof_{col}" for col in info_prof_expanded.columns]
            merged_df = pd.concat([merged_df.drop(columns=['informacoes_profissionais']), info_prof_expanded], axis=1)
        except Exception as e:
            print(f"Erro ao normalizar 'informacoes_profissionais': {e}")

    if 'formacao_e_idiomas' in merged_df.columns:
        try:
            form_idiomas_expanded = pd.json_normalize(merged_df['formacao_e_idiomas'].fillna({}))
            form_idiomas_expanded.columns = [f"app_form_{col}" for col in form_idiomas_expanded.columns]
            merged_df = pd.concat([merged_df.drop(columns=['formacao_e_idiomas']), form_idiomas_expanded], axis=1)
        except Exception as e:
            print(f"Erro ao normalizar 'formacao_e_idiomas': {e}")
    # Salvar o DataFrame processado em pickle
    output_path = '../data/processed_data.pkl'
    merged_df.to_pickle(output_path)
    print(f"Dados processados e salvos em {output_path}")
    print(f"Shape do DataFrame salvo: {merged_df.shape}")
    print(f"Algumas colunas do DataFrame salvo: {merged_df.columns.tolist()[:20]}...")


if __name__ == '__main__':
    prepare_data()

