# Dockerfile
FROM python:3.12

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo requirements.txt
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código fonte
COPY src/ /app/src

# Exponha a porta da aplicação
EXPOSE 5000

# Comando para iniciar a aplicação
CMD ["python", "src/main.py"]
