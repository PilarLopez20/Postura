# Descargar la imafen de ubuntu

FROM ubuntu:22.04

#Actualizar la lista de imagen 
RUN apt-get update && apt-get upgrade -y

#Actualizar la imagen 

RUN apt-get install -y python3 python3-pip

#Instalar herramientas

RUN apt-get install python3 -y

#Copiar la carpeta webapp
# Establece el directorio de trabajo
WORKDIR /main

# Copia los archivos del proyecto al contenedor
COPY . .

# Actualiza pip  
RUN pip install --upgrade pip

#Instalar las librerias
RUN pip install -r requirements.txt

# Comando para iniciar la aplicación
EXPOSE 5000
CMD ["sh", "-c", "gunicorn -w 4 -b 0.0.0.0:${PORT:-5000} main:app"]
