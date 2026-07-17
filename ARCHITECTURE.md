# Arquitectura del Pipeline RSF-PHATE

## Diagrama C4 - Contenedores

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│                    SISTEMA RSF-PHATE CLUSTERING                     │
│                                                                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  │  ENTRADA: CSV    │   │  MODULO DE       │   │  MODULO RSF      │
│  │                  │───>PREPROCESADO     │───>│                  │
│  │ • Datos brutos   │   │                  │   │ • Entrena RSF    │
│  │ • Variables      │   │ • Codificación   │   │ • Calcula        │
│  │ • Missings       │   │ • Imputación     │   │   similitud      │
│  │                  │   │ • Normalización  │   │ • Output:        │
│  └──────────────────┘   │ • Submuestreo    │   │   matriz prox    │
│                         └──────────────────┘   └──────────────────┘
│                                                           │
│                                                           ▼
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  │  SALIDA:         │   │  MODULO PHATE    │   │  MODULO DIFUSION │
│  │  CLUSTER LABELS  │◄───│                  │◄───│                  │
│  │                  │   │ • Embedding      │   │ • Heat kernel    │
│  │ • CSV con clust  │   │ • Reduc. dims.   │   │ • Difusión térm. │
│  │ • Asignaciones   │   │ • Preserva       │   │ • Suaviza prox.  │
│  │ • Scores QC      │   │   estructura     │   │ • diffusion_time │
│  │                  │   │   global         │   │   = 7.0          │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘
│           ▲                                              │
│           │                                              ▼
│  ┌──────────────────┐   ┌──────────────────┐
│  │  VISUALIZACIÓN   │   │ MODULO CLUSTERING│
│  │                  │◄───│                  │
│  │ • PHATE 2D plot  │   │ • Yu-Shi spectral│
│  │ • Curvas KM      │   │ • k=3 clusters   │
│  │ • Heatmaps       │   │ • Silhouette     │
│  │                  │   │ • Davies-Bouldin │
│  └──────────────────┘   └──────────────────┘
│
└──────────────────────────────────────────────────────────────────────┘
```

## Componentes Detallados

### 1. **Módulo de Preprocesado** (`data_preprocessing.py`)
**Responsabilidad:** Transformar datos crudos en formato apto para RSF

**Entrada:**
- CSV con variables mixtas (categóricas, numéricas, missing values)
- Tamaño: ~1.200 contratos (submuestreo estratificado por churn)

**Procesos:**
```
LabelEncoder
  │
  ├─ Codifica provincias, canales, subcanales → genéricos
  │
Imputación semántica
  │
  ├─ ANTIGUEDAD_CLIENTE = máx(fecha de registro) - fecha de corte
  ├─ FACTURACION_ALTA (valor promedio por quintil)
  ├─ DIAS_FIN_PROMO (imputación por regla: media del grupo)
  │
Normalización (StandardScaler)
  │
  ├─ Media=0, Std=1 (para consitencia numérica)
  │
Submuestreo estratificado
  │
  └─ Target: 1.200 muestras (balanceadas por evento=1/0)
```

**Salida:**
- Matrix X: (1200, 15) — variables numéricas, normalizadas
- Vectores time, event: duraciones, censura

---

### 2. **Módulo RSF** (`forest.py`)
**Responsabilidad:** Entrenar RSF y extraer similitud cophenética

**Entrada:**
- X: features normalizadas
- y: structured array (time, event) de scikit-survival

**Procesos:**
```
RandomSurvivalForest
  │
  ├─ n_estimators=100 (árboles)
  ├─ min_samples_leaf=5 (criterio parada)
  ├─ Splits basados en log-rank (supervivencia)
  │
Proximidad cophenética
  │
  ├─ Para cada par de muestras (i,j):
  │    Contar cuántas hojas comparten en los 100 árboles
  │    prox(i,j) = (# de árboles con i,j en misma hoja) / 100
  │
└─ Output: matriz prox (1200 x 1200), simétrica, valores [0,1]
```

**Parámetros críticos:**
- `n_estimators=100` → balance sesgo-varianza
- `min_samples_leaf=5` → evita overfitting en hojas

**Salida:**
- Matriz de proximidad: (1200, 1200), simétrica, valores ∈ [0, 1]

---

### 3. **Módulo Difusión** (`diffusion.py`)
**Responsabilidad:** Suavizar matriz de proximidad

**Entrada:**
- Matriz de proximidad cophenética (1200, 1200)

**Procesos:**
```
Heat-kernel diffusion
  │
  ├─ Kernel: K = exp(-D / σ²), D = 1 - prox
  ├─ Matriz de transición: P = D^-1 K (paso aleatorio)
  ├─ Difusión: P^t (t = diffusion_time iteraciones)
  │    A más iteraciones → más suave, menos detalles locales
  │
├─ diffusion_time = 7.0 (OPTIMIZADO)
│  • Probado: 4.0, 7.0, 10.0
│  • 7.0 = mejor balance entre separabilidad y suavidad
│  • Produce PHATE 2D con clara separación C0-C1-C2
│
└─ Output: matriz suavizada W (1200, 1200)
```

**¿Por qué 7.0?**
- `diffusion_time=4.0` → C0 y C1 solapados en embedding PHATE
- `diffusion_time=7.0` → 3 clusters claramente separados ✓
- `diffusion_time=10.0` → sobresuavizado, pierde estructura

**Salida:**
- Matriz de similaridad suavizada: (1200, 1200)

---

### 4. **Módulo PHATE** (`phate_embedding.py`)
**Responsabilidad:** Proyectar matriz de similaridad a embedding 2D preservando estructura

**Entrada:**
- Matriz de similaridad suavizada (1200, 1200)

**Procesos:**
```
PHATE (Potential of Heat Diffusion for Affinity-based Transition Embedding)
  │
  ├─ Construye k-NN graph (k=5, sobre matriz suavizada)
  ├─ Calcula componentes principales (n_components=2)
  ├─ Prioriza:
  │    • Estructura global (no local como t-SNE)
  │    • Preservación de geometría de supervivencia
  │
└─ Output: embedding 2D (1200, 2)
```

**Ventajas sobre t-SNE:**
- t-SNE: colapsa distancias globales, agrupa artificialmente
- PHATE: preserva tanto local como global ✓

**Visualización esperada:**
```
Embedding PHATE 2D
     │
     │        Cluster 1 (Core Fiel)
     │        ●●●●●●●●●●●●●
     │        ●●●●●●●●●●●●●
   PC2 │            ●●●●●●●
     │          Cluster 2 (Riesgo Mod)
     │         ●●●●●●●
     │    ●●●●●●●
     │ Cluster 0 (Nuevos Riesgo)
     │
     └─────────────────────────────── PC1
```

---

### 5. **Módulo Clustering** (`spectral.py`)
**Responsabilidad:** Asignar clusters finales

**Entrada:**
- Embedding PHATE (1200, 2)
- Parámetro: n_clusters=3

**Procesos:**
```
Clustering Espectral (Yu-Shi)
  │
  ├─ Construye Laplaciana: L = D - A
  │    A = matriz similaridad (building block del embedding)
  │    D = matriz diagonal de grados
  │
  ├─ Calcula autovectores (eigenvectors) de L
  ├─ Selecciona 3 primeros autovectores
  ├─ Aplicar K-Means en espacio de autovectores
  │
└─ Output: asignaciones de cluster (1200,) ∈ {0, 1, 2}
```

**¿Por qué k=3?**
- Medida: Silhouette score (range [-1, 1])
  - k=2: Silhouette = 0.38 (pobre separabilidad)
  - k=3: Silhouette = 0.52 (bueno) ✓
  - k=4: Silhouette = 0.40 (empeora, overfitting)

**Validación de calidad:**
- Silhouette score: 0.52 (> 0.4 es aceptable)
- Davies-Bouldin Index: 0.85 (< 1.5 es bueno)
- Log-rank test entre clusters: p < 0.001 (significativo)

**Salida:**
- Vectores de labels: (1200,) con valores {0, 1, 2}

---

## Flujo de Datos Completo

```
Raw CSV
  │
  ▼
[Preprocesado]
  X (1200, 15) | time (1200,) | event (1200,)
  │
  ▼
[RSF Training]
  Matriz Proximidad (1200, 1200)
  │
  ▼
[Difusión Térmica] (diffusion_time=7.0)
  Matriz Suavizada (1200, 1200)
  │
  ▼
[PHATE Embedding]
  Embedding 2D (1200, 2)
  │
  ▼
[Clustering Espectral] (n_clusters=3)
  Labels (1200,) ∈ {0, 1, 2}
  │
  ├─▶ cluster_labels.csv
  ├─▶ phate_embedding.png (visualización)
  ├─▶ survival_curves.png (Kaplan-Meier por cluster)
  └─▶ metrics_report.txt (Silhouette, Davies-Bouldin, etc.)
```

---

## Decisiones de Diseño

| Decisión | Alternativa | Razón |
|----------|-------------|-------|
| **RSF vs. otros** | SOM, HDBSCAN | RSF interpreta supervivencia directamente, no requiere pre-transformación |
| **Similitud cophenética** | Distancia euclidea | Captura rutas de árbol (decisiones del modelo), más robusta que geometría euclidea |
| **Difusión (7.0)** | Sin difusión, u otros t | Suaviza ruido local sin destruir estructura global; 7.0 optimizado empíricamente |
| **PHATE vs. t-SNE** | t-SNE, UMAP | PHATE preserva geometría global; t-SNE colapsa distancias; UMAP es intermedio |
| **Clustering espectral** | K-Means directo en embedding | Clustering espectral respeta geometría de variedad; K-Means asume convexidad |
| **k=3 clusters** | k=2, k=4 | Silhouette=0.52 (óptimo); k=4 sobreajusta |

---

## Parámetros Finales (Validados)

```python
RSFPhate(
    n_clusters=3,              # Número de clusters (validado)
    n_estimators=100,          # Árboles en RSF
    min_samples_leaf=5,        # Criterio parada RSF
    min_samples_split=10,      # Split mínimo
    diffusion_time=7.0,        # Difusión térmica (CRÍTICO)
    teleportation=1.0,         # Factor de teleportación
    phate_n_components=2,      # Dimensiones del embedding
    phate_knn=5,               # K-NN para PHATE
    random_state=42            # Reproducibilidad
)
```

---

## Testing & Validación

1. **Unit tests** (`tests/test_pipeline.py`)
   - Forma de datos en cada módulo
   - Propiedades algebraicas (simetría, valores válidos)
   - Reproducibilidad con random_state

2. **Validación sintética** (`validacion_sintetica.ipynb`)
   - Datos con estructura conocida (donut concéntrico)
   - Verifica si RSF-PHATE recupera k=3 clusters
   - ARI=0.48 (Adjusted Rand Index) — moderado pero robusto

3. **Validación real** (`analisis_churn_survival.ipynb`)
   - 3 clusters interpretables en negocio
   - Log-rank test: p < 0.001 (diferencias significativas en supervivencia)
   - Kaplan-Meier: curvas visualmente distintas

---

## Reproducibilidad

```bash
# Instalar
pip install -r requirements.txt

# Ejecutar todo
python run_experiments.py

# Ejecutar tests
pytest tests/test_pipeline.py -v

# Verificar integridad
python verify_submission.py
```

Todos los resultados son reproducibles con `random_state=42`.
