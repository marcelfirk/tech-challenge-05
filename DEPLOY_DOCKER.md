# Documentação para Deploy da API de Recrutamento com Docker

Este documento descreve como construir e executar a API de recrutamento em um container Docker.

## Pré-requisitos

- Docker instalado e em execução no seu ambiente.

## Arquivos Necessários

Certifique-se de que os seguintes arquivos estejam no diretório raiz do projeto (`recruitment_api`):

- `Dockerfile`: Contém as instruções para construir a imagem Docker.
- `requirements.txt`: Lista todas as dependências Python necessárias para a API.
- `src/`: Diretório contendo o código-fonte da aplicação Flask (`main.py`, `routes/prediction.py`, etc.).
- `model_rf.joblib`: O arquivo do modelo de Machine Learning treinado (Random Forest). Este arquivo deve estar em `/home/ubuntu/model_rf.joblib` no seu sistema local para que o `Dockerfile` possa copiá-lo para o local correto dentro da imagem, ou você pode ajustar o comando `COPY` no `Dockerfile` se o seu modelo estiver em outro local.

**Observação sobre o caminho do modelo:** O `Dockerfile` atual está configurado para copiar o modelo de `/home/ubuntu/model_rf.joblib` (do host) para `/home/ubuntu/model_rf.joblib` (dentro do container). O script `src/routes/prediction.py` espera carregar o modelo deste caminho dentro do container. Se o seu arquivo `model_rf.joblib` estiver em um local diferente no seu sistema host, ajuste a linha `COPY /home/ubuntu/model_rf.joblib /home/ubuntu/model_rf.joblib` no `Dockerfile` para o caminho correto.

## Construindo a Imagem Docker

1.  Navegue até o diretório raiz do projeto `recruitment_api` no seu terminal.
2.  Execute o seguinte comando para construir a imagem Docker:

    ```bash
    docker build -t recruitment-api .
    ```

    Isso criará uma imagem Docker chamada `recruitment-api`.

## Executando o Container Docker

1.  Após a construção bem-sucedida da imagem, execute o seguinte comando para iniciar um container a partir da imagem:

    ```bash
    docker run -p 5000:5000 recruitment-api
    ```

    -   `-p 5000:5000`: Mapeia a porta 5000 do container para a porta 5000 do seu host. A API Flask estará acessível em `http://localhost:5000`.
    -   `recruitment-api`: O nome da imagem Docker que você construiu.

2.  A API Flask deverá estar rodando dentro do container. Você verá logs no terminal indicando que o servidor Flask foi iniciado.

## Testando a API em Execução no Docker

Após iniciar o container, você pode testar o endpoint `/predict` usando uma ferramenta como cURL ou Postman. Exemplo com cURL (supondo que você tenha um arquivo `test_payload.json` com os dados de entrada no formato correto):

```bash
curl -X POST -H "Content-Type: application/json" -d @test_payload.json http://localhost:5000/predict
```

O `test_payload.json` deve conter os campos esperados pelo modelo, conforme definido em `src/routes/prediction.py` (variável `EXPECTED_FEATURES`).

## Estrutura do Projeto Esperada para o Build Docker

```
recruitment_api/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── routes/
│       ├── __init__.py
│       └── prediction.py
└── (outros arquivos como venv/ que não são copiados para o Docker)

# O modelo model_rf.joblib deve estar acessível para o comando COPY no Dockerfile
# Por exemplo, em /home/ubuntu/model_rf.joblib no sistema que executa o docker build
```

## Solução de Problemas

-   **Erro `FileNotFoundError` para `model_rf.joblib` dentro do container:** Verifique se o caminho no comando `COPY` do `Dockerfile` para `model_rf.joblib` está correto e se o arquivo existe nesse local no momento do build. Além disso, confirme se o caminho `MODEL_PATH` em `src/routes/prediction.py` corresponde ao local onde o modelo é copiado dentro do container (`/home/ubuntu/model_rf.joblib`).
-   **Erro de dependências:** Certifique-se de que o arquivo `requirements.txt` está completo e atualizado com todas as bibliotecas necessárias (Flask, joblib, scikit-learn, pandas, numpy). Execute `pip3 freeze > requirements.txt` dentro do ambiente virtual (`venv`) do projeto antes de construir a imagem Docker.


