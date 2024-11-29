FROM debian:bullseye-slim

# Establece el directorio de trabajo
WORKDIR /app

# Instala Python, pip y las dependencias necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos del proyecto al contenedor
COPY . .

# Actualiza pip
RUN pip3 install --upgrade pip

# Instala las dependencias del proyecto
RUN pip3 install --no-cache-dir -r requirements.txt

# Comando para iniciar la aplicación
CMD ["sh", "-c", "gunicorn -w 4 -b 0.0.0.0:${PORT:-5000} app:app"]
