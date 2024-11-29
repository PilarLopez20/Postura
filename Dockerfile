FROM debian:bullseye-slim

WORKDIR /app
# Instalar Python, pip y las dependencias necesarias
RUN apk add --no-cache python3 py3-pip
COPY . .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["sh", "-c", "gunicorn -w 4 -b 0.0.0.0:${PORT:-5000} app:app"]

