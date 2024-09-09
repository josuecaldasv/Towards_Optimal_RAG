# Utilizar una imagen base con Python 3.12
FROM python:3.12-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY . /app

# Instalar los requisitos del proyecto
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Instalar Jupyter Notebook
RUN pip install jupyter

# Exponer el puerto para Jupyter Notebook
EXPOSE 10002

# Comando para iniciar Jupyter Notebook al ejecutar el contenedor
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--port=10002"]