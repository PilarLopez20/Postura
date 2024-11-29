FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /main

# Instala las dependencias necesarias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos del proyecto al contenedor
COPY . .

# Actualiza pip
RUN pip install --upgrade pip

# Instala las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Comando para iniciar la aplicación
CMD ["sh", "-c", "gunicorn -w 4 -b 0.0.0.0:${PORT:-5000} app:app"]
