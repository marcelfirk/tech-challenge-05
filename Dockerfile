# Use uma imagem base oficial do Python.
FROM python:3.11-slim

# Defina o diretório de trabalho no container.
WORKDIR /app

# Copie o arquivo de dependências primeiro para aproveitar o cache do Docker.
COPY requirements.txt requirements.txt

# Instale as dependências.
# É importante garantir que todas as dependências, incluindo as de ML, estejam aqui.
RUN pip3 install --no-cache-dir -r requirements.txt

# Copie o diretório src da API para o diretório de trabalho.
COPY src/ ./src/

# Copie o modelo treinado para dentro do container.
# Certifique-se de que o caminho em prediction.py corresponda a este local no container ou ajuste o caminho.
# No prediction.py, o modelo é carregado de "/home/ubuntu/model_rf.joblib".
# Para simplificar, vamos copiar o modelo para a raiz do app e ajustar o prediction.py ou 
# criar um diretório /home/ubuntu/ no container e copiar para lá.
# Optaremos por copiar para /app/model_rf.joblib e ajustar o prediction.py depois, se necessário,
# ou melhor, o prediction.py já espera o modelo em um caminho absoluto. Vamos manter o caminho absoluto
# e garantir que o modelo seja copiado para o local esperado pelo script prediction.py dentro do container.
# O script prediction.py espera o modelo em /home/ubuntu/model_rf.joblib
# Então, criaremos esse diretório no container e copiaremos o modelo para lá.
RUN mkdir -p /home/ubuntu
COPY /home/ubuntu/model_rf.joblib /home/ubuntu/model_rf.joblib

# Exponha a porta em que o aplicativo Flask será executado.
EXPOSE 5000

# Defina a variável de ambiente para o Flask (opcional, mas bom para produção).
ENV FLASK_APP=src/main.py
ENV FLASK_RUN_HOST=0.0.0.0

# Comando para executar o aplicativo quando o container iniciar.
# Usamos python3 diretamente, pois é o padrão na imagem python:3.11-slim.
CMD ["python3", "src/main.py"]

