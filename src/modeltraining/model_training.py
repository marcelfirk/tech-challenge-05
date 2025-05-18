import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import numpy as np

def train_model():
    print("Iniciando o treinamento do modelo...")
    try:
        df = pd.read_pickle("/home/ubuntu/processed_data.pkl")
        print(f"DataFrame carregado com shape: {df.shape}")
    except Exception as e:
        print(f"Erro ao carregar /home/ubuntu/processed_data.pkl: {e}")
        return

    # Reduzir o dataset para testes iniciais e evitar problemas de memória/tempo
    # df = df.sample(n=5000, random_state=42) if len(df) > 5000 else df
    # print(f"Dataset reduzido para {len(df)} amostras para teste.")

    # Definir X e y
    if 'target' not in df.columns:
        print("Coluna 'target' não encontrada no DataFrame.")
        return
    
    X = df.drop(columns=["target", "ID_VAGA", "ID_APPLICANT", "situacao_candidado", "titulo_vaga_prospect", 
                         "informacoes_basicas", "beneficios", "ID_APPLICANT_raw", "infos_basicas_applicant", 
                         "informacoes_pessoais", "cargo_atual", "cv_en"])
    y = df["target"]

    # Garantir que não há NaNs em y antes do split
    nan_in_y_before = y.isnull().sum()
    if nan_in_y_before > 0:
        print(f"Encontrados {nan_in_y_before} NaNs na variável alvo 'y' antes da limpeza.")
        # Remover NaNs de y e alinhar X
        X = X[~y.isnull()]
        y = y.dropna()
        print(f"NaNs removidos de 'y'. Novo shape de X: {X.shape}, novo shape de y: {y.shape}")
    else:
        print("Nenhum NaN encontrado na variável alvo 'y' antes do split.")

    print(f"Shape de X: {X.shape}, Shape de y: {y.shape}")

    # Identificar tipos de colunas para pré-processamento
    text_features = []
    categorical_features = []
    numerical_features = [] # Não temos numéricas explícitas além do que será gerado por TF-IDF

    # Colunas de texto principais (CV e descrição da vaga)
    if 'cv_pt' in X.columns:
        text_features.append('cv_pt')
        X['cv_pt'] = X['cv_pt'].fillna('') # Preencher NaNs em colunas de texto
    if 'vaga_principais_atividades' in X.columns:
        text_features.append('vaga_principais_atividades')
        X['vaga_principais_atividades'] = X['vaga_principais_atividades'].fillna('')
    if 'vaga_competencia_tecnicas_e_comportamentais' in X.columns:
        text_features.append('vaga_competencia_tecnicas_e_comportamentais')
        X['vaga_competencia_tecnicas_e_comportamentais'] = X['vaga_competencia_tecnicas_e_comportamentais'].fillna('')
    if 'app_prof_conhecimentos_tecnicos' in X.columns:
        text_features.append('app_prof_conhecimentos_tecnicos')
        X['app_prof_conhecimentos_tecnicos'] = X['app_prof_conhecimentos_tecnicos'].fillna('')

    # Colunas categóricas (exemplos, adicionar mais conforme a análise)
    potential_cat_cols = [
        'vaga_nivel profissional', 'vaga_nivel_academico', 'vaga_nivel_ingles', 'vaga_nivel_espanhol',
        'vaga_areas_atuacao', 'vaga_local_trabalho', 'vaga_vaga_especifica_para_pcd',
        'app_prof_nivel_profissional', 'app_prof_area_atuacao', 
        'app_form_nivel_academico', 'app_form_nivel_ingles', 'app_form_nivel_espanhol'
    ]

    for col in potential_cat_cols:
        if col in X.columns:
            X[col] = X[col].fillna('Desconhecido').astype(str) # Preencher NaNs e garantir tipo string
            if X[col].nunique() < 50: # Limitar cardinalidade para OneHotEncoding
                 categorical_features.append(col)
            # else: # Para alta cardinalidade, poderia usar Target Encoding ou Embedding, ou TF-IDF se for texto livre
            #    print(f"Coluna {col} tem alta cardinalidade ({X[col].nunique()}) e será tratada como texto ou descartada.")
            #    X[col] = X[col].fillna('') 
            #    text_features.append(col) # Tratar como texto se alta cardinalidade

    print(f"Text features: {text_features}")
    print(f"Categorical features: {categorical_features}")

    # Manter apenas as colunas que serão usadas
    features_to_keep = list(set(text_features + categorical_features))
    if not features_to_keep:
        print("Nenhuma feature selecionada para o modelo. Verifique a lógica de seleção.")
        return
    X = X[features_to_keep]
    print(f"Shape de X após seleção de features: {X.shape}")
    print(f"Colunas finais em X: {X.columns.tolist()}")

    # Criar preprocessor
    preprocessor_steps = []
    if text_features:
        for i, feature_name in enumerate(text_features):
            preprocessor_steps.append((f'tfidf_{i}', TfidfVectorizer(max_features=1000, stop_words=None, ngram_range=(1,2)), feature_name))
    
    if categorical_features:
        preprocessor_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features))

    if not preprocessor_steps:
        print("Nenhum passo de pré-processamento definido. Verifique as features.")
        return
        
    preprocessor = ColumnTransformer(transformers=preprocessor_steps, remainder='drop') # drop other columns

    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Dados divididos: X_train: {X_train.shape}, X_test: {X_test.shape}")

    # --- Modelo 1: Logistic Regression ---
    print("\n--- Treinando Logistic Regression ---")
    pipeline_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler(with_mean=False)), # StandardScaler para dados esparsos
        ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', C=0.1))
    ])
    
    try:
        pipeline_lr.fit(X_train, y_train)
        y_pred_lr = pipeline_lr.predict(X_test)
        y_proba_lr = pipeline_lr.predict_proba(X_test)[:, 1]

        print("Logistic Regression - Relatório de Classificação:")
        print(classification_report(y_test, y_pred_lr))
        print(f"Logistic Regression - ROC AUC: {roc_auc_score(y_test, y_proba_lr)}")
        print(f"Logistic Regression - Accuracy: {accuracy_score(y_test, y_pred_lr)}")
        joblib.dump(pipeline_lr, '/home/ubuntu/model_lr.joblib')
        print("Modelo Logistic Regression salvo em /home/ubuntu/model_lr.joblib")

    except Exception as e:
        print(f"Erro ao treinar Logistic Regression: {e}")
        import traceback
        traceback.print_exc()

    # --- Modelo 2: Random Forest (mais robusto, mas pode ser mais lento) ---
    print("\n--- Treinando Random Forest ---")
    pipeline_rf = Pipeline([
        ('preprocessor', preprocessor),
        # RandomForest não necessita de scaling, mas pode ser útil para consistência ou se outros modelos forem adicionados
        # ('scaler', StandardScaler(with_mean=False)), 
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10, n_jobs=-1))
    ])

    # Parâmetros para GridSearchCV (opcional, para otimizar)
    # param_grid_rf = {
    #     'classifier__n_estimators': [50, 100],
    #     'classifier__max_depth': [10, 20, None]
    # }
    # grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=2, scoring='roc_auc', verbose=1, n_jobs=-1)

    try:
        # grid_search_rf.fit(X_train, y_train)
        # print(f"Melhores parâmetros para Random Forest: {grid_search_rf.best_params_}")
        # best_pipeline_rf = grid_search_rf.best_estimator_
        
        pipeline_rf.fit(X_train, y_train)
        best_pipeline_rf = pipeline_rf # Usar o pipeline treinado diretamente sem gridsearch por agora

        y_pred_rf = best_pipeline_rf.predict(X_test)
        y_proba_rf = best_pipeline_rf.predict_proba(X_test)[:, 1]

        print("Random Forest - Relatório de Classificação:")
        print(classification_report(y_test, y_pred_rf))
        print(f"Random Forest - ROC AUC: {roc_auc_score(y_test, y_proba_rf)}")
        print(f"Random Forest - Accuracy: {accuracy_score(y_test, y_pred_rf)}")
        joblib.dump(best_pipeline_rf, '/home/ubuntu/model_rf.joblib')
        print("Modelo Random Forest salvo em /home/ubuntu/model_rf.joblib")

    except Exception as e:
        print(f"Erro ao treinar Random Forest: {e}")
        import traceback
        traceback.print_exc()

    print("\nTreinamento e avaliação concluídos. Modelos salvos.")

if __name__ == '__main__':
    train_model()

