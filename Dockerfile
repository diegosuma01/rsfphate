# =============================================================================
# Imagen reproducible del pipeline RSF-PHATE (TFG Diego Suárez Marañón)
# Construir:  docker build -t rsfphate .
# Ejecutar:   docker run --rm -v ${PWD}/data:/app/data rsfphate
# =============================================================================
FROM python:3.11-slim

# Dependencias del sistema necesarias para scikit-survival / compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias primero (aprovecha la caché de capas)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar el código y la configuración
COPY . .
RUN pip install --no-cache-dir -e .

# Reproduce todos los experimentos de la memoria
CMD ["python", "run_experiments.py"]
