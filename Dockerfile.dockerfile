FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Por defecto, el comando es iniciar el proceso ETL
CMD ["make", "etl"]
