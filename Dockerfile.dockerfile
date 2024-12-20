# Dockerfile

# Usar una imagen base de Python
FROM python:3.11-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo requirements.txt e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar DVC
RUN pip install dvc

# Copiar todo el código del proyecto al contenedor
COPY . .

# Comando predeterminado para ejecutar el pipeline ETL
CMD ["make", "clean"]
