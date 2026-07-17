"""
Preprocesado del dataset anonimizado de contratos de energía eléctrica
para análisis de supervivencia con RSF-PHATE.

¿Por qué "análisis de supervivencia"?
--------------------------------------
En lugar de predecir SI un cliente va a hacer churn, el análisis de
supervivencia nos pregunta: ¿CUÁNTO TIEMPO llevará activo antes de darse
de baja? Esto nos permite:

  - Descubrir grupos de clientes con patrones de churn distintos (clustering).
  - Entender qué características se asocian a una baja "temprana" vs "tardía".
  - Manejar correctamente a los clientes que AÚN NO han hecho churn (censura
    a la derecha): sabemos que llevan K meses activos, pero no cuándo se irán.

Las tres variables clave del modelo de supervivencia son:
  ┌──────────┬─────────────────────────┬─────────────────────────────────────┐
  │ Variable │ Columna en el CSV       │ Significado                         │
  ├──────────┼─────────────────────────┼─────────────────────────────────────┤
  │ time     │ ANTIGUEDAD_CLIENTE      │ Meses activo hasta baja o corte     │
  │ event    │ TARGET (bool)           │ True = se dio de baja (churn real)  │
  │ X        │ resto de columnas       │ Covariables del contrato            │
  └──────────┴─────────────────────────┴─────────────────────────────────────┘
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ── Columnas categóricas que codificamos numéricamente ────────────────────────
# Son variables de texto/categoría que el modelo no puede usar directamente.
CATEGORICAS = [
    'CANAL_INPUT',             # canal de captación (CANAL 1, CANAL 2, ...)
    'SUBCANAL_INPUT',          # subcanal de captación
    'PROVINCIA_PS',            # provincia del punto de suministro (anonimizada)
    'RENOVACION_CAPTACION_MOD',# tipo de renovación/captación
    'GEN_MOD',                 # variable generada/calculada de tipo categórico
    'IND_Alta_Adva',           # indicador de alta administrativa
]

# ── Columnas que descartamos completamente ────────────────────────────────────
# Motivo de cada descarte (mismo criterio que en preprocesado.py original):
IGNORAR = [
    'NUM_CONTRATO',             # identificador de contrato (no es feature predictivo)
    'PROVINCIA_AGR',            # redundante con PROVINCIA_PS
    'FECHA_ALTA_INSTALACION',   # la info útil ya está en ANTIGUEDAD_CLIENTE
    'FECHA_BAJA_INSTALACION',   # la info útil ya está en TARGET + ANTIGUEDAD_CLIENTE
    'COD_SEXO',                 # muchos NaN, bajo valor predictivo
    'ANTIGUEDAD',               # redundante con ANTIGUEDAD_CLIENTE (versión alternativa)
    'CHURN_COD_POSTAL',         # proxied por PROVINCIA_PS
    'ANTIG_CLI_RANGO',          # versión categórica/rango de ANTIGUEDAD_CLIENTE
    'DIAS_FIN_PROMO_RANGO',     # versión rango de DIAS_FIN_PROMO (redundante)
    'N_PEDIDOINFO_FECHA_REF_3_BIN',  # versión binaria de la variable MOD (redundante)
    'N_CONTACTO_FECHA_REF_1',   # referencia temporal antigua, poco informativa
    'N_CONTACTO_FECHA_REF_2',   # ídem
    'DIAS_PARA_RENOVAR',        # redundante con DIAS_PARA_RENOVAR_MOD (versión corregida)
                                # además tiene ~58% de NaN → causaría pérdida masiva de filas
    'DURACION_PROMO',           # solo en ABT_ELECTRICIDAD, no en anonimizado
    'DURACION_PROMO_RANGO',     # ídem
    'NOMBRE_IC',                # nombre de cliente (solo en ABT_ELECTRICIDAD)
]


def cargar_datos(ruta: str, encoding: str = 'iso-8859-1') -> pd.DataFrame:
    """
    Lee el CSV anonimizado y muestra un resumen básico.

    Parameters
    ----------
    ruta     : ruta absoluta o relativa al fichero CSV
    encoding : codificación del fichero (el original usa iso-8859-1)

    Returns
    -------
    DataFrame con todos los datos sin modificar
    """
    df = pd.read_csv(ruta, encoding=encoding)

    n_total = len(df)
    n_churn = int(df['TARGET'].sum())
    print(f"Datos cargados:        {n_total:>9,} filas × {df.shape[1]} columnas")
    print(f"Clientes con churn:    {n_churn:>9,}  ({n_churn / n_total:.2%})")
    print(f"Clientes sin churn:    {n_total - n_churn:>9,}  ({1 - n_churn/n_total:.2%})")
    return df


def preparar_muestra_supervivencia(
    df: pd.DataFrame,
    n_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Extrae una muestra ESTRATIFICADA del dataset completo.

    ¿Por qué submuestrear?
    RSF-PHATE construye una matriz de proximidad de tamaño n×n.
    Con 678K filas, esa matriz ocuparía ~3.5 TB de RAM.
    La complejidad es O(n²), así que reducir de 678K a 1000 filas
    hace el cálculo millones de veces más ligero.

    La estratificación por TARGET conserva la tasa de churn original
    para que la muestra sea representativa.

    Parameters
    ----------
    df         : DataFrame completo
    n_samples  : tamaño de la muestra (recomendado: 1000–2000 para demos)
    random_state : semilla de aleatoriedad para reproducibilidad

    Returns
    -------
    DataFrame submuestreado con la misma proporción de churn que el original
    """
    churn_rate_original = df['TARGET'].mean()
    frac = n_samples / len(df)

    # groupby + apply mantiene la proporción de TARGET=0 y TARGET=1
    sample = (
        df.groupby('TARGET', group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=random_state))
        .reset_index(drop=True)
    )

    n_sample = len(sample)
    churn_rate_sample = sample['TARGET'].mean()
    print(f"Muestra generada:      {n_sample:>6,} filas")
    print(f"Churn rate original:   {churn_rate_original:.2%}")
    print(f"Churn rate en muestra: {churn_rate_sample:.2%}")
    return sample


def preprocesar_supervivencia(df: pd.DataFrame):
    """
    Transforma el (sub)dataset al formato que espera RSFPhate.

    Pasos internos:
      1. Convierte fechas (para luego descartarlas limpiamente)
      2. Extrae 'time' (ANTIGUEDAD_CLIENTE) y 'event' (TARGET) como variables
         de supervivencia separadas
      3. Determina qué columnas usar como covariables (X)
      4. Codifica variables categóricas con LabelEncoder
      5. Rellena NaN en columnas concretas con 0 (igual que preprocesado.py)
      6. Elimina filas con NaN restantes
      7. Convierte a float32 para reducir uso de memoria

    Parameters
    ----------
    df : DataFrame (ya submuestreado o completo)

    Returns
    -------
    X     : pd.DataFrame de float32  — covariables del contrato
    time  : pd.Series de float32     — ANTIGUEDAD_CLIENTE (días activo)
    event : pd.Series de bool        — True si el cliente churneó (TARGET=1)
    """
    df = df.copy()

    # ── Paso 1: convertir fechas ──────────────────────────────────────────────
    # Solo las convertimos para que el drop posterior sea limpio;
    # no las usamos directamente (ANTIGUEDAD_CLIENTE ya recoge esa info).
    for col in ['FECHA_ALTA_INSTALACION', 'FECHA_BAJA_INSTALACION']:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col], format="%d%b%Y:%H:%M:%S", errors='coerce'
            )

    # ── Paso 2: separar las variables de supervivencia ────────────────────────
    # Estas columnas NO irán a X; son las "respuestas" del modelo.
    time = df['ANTIGUEDAD_CLIENTE'].copy()   # tiempo de supervivencia (meses)
    event = df['TARGET'].astype(bool).copy() # True = churn observado

    # ── Paso 3: definir las covariables X ────────────────────────────────────
    # Excluimos: columnas de supervivencia + lista IGNORAR
    # (algunas columnas de IGNORAR pueden no existir en el anonimizado → seguro)
    excluir = set(IGNORAR + ['TARGET', 'ANTIGUEDAD_CLIENTE'])
    excluir = [c for c in excluir if c in df.columns]

    columnas_features = [c for c in df.columns if c not in excluir]
    df_feat = df[columnas_features].copy()

    # ── Paso 4: codificar categóricas ────────────────────────────────────────
    # LabelEncoder convierte strings como "CANAL 1", "CANAL 2" → 0, 1, 2, ...
    # Se re-ajusta en cada columna por separado (no hay leakage entre columnas).
    le = LabelEncoder()
    for col in CATEGORICAS:
        if col in df_feat.columns:
            df_feat[col] = le.fit_transform(df_feat[col].astype(str))

    # ── Paso 5: rellenar NaN conocidos con 0 ─────────────────────────────────
    # Estos NaN significan "no aplica" (ej. sin promoción activa → DIAS_FIN_PROMO=0)
    fill_zeros = {
        'DIAS_PARA_RENOVAR_MOD': 0,  # NaN = no hay fecha de renovación → 0
        'DIAS_FIN_PROMO':        0,  # NaN = no está en promoción → 0
        'N_CONTACTO_FECHA_REF_3':0,  # NaN = sin contactos registrados → 0
    }
    for col, val in fill_zeros.items():
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].fillna(val)

    # ── Paso 6: eliminar filas con NaN restantes ──────────────────────────────
    valid = df_feat.notna().all(axis=1) & time.notna() & event.notna()
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"  → {n_dropped} filas eliminadas por NaN")

    df_feat = df_feat[valid].reset_index(drop=True)
    time    = time[valid].reset_index(drop=True)
    event   = event[valid].reset_index(drop=True)

    # ── Paso 7: convertir a float32 ────────────────────────────────────────
    # float32 usa la mitad de RAM que float64, importante para matrices n×n.
    X = df_feat.astype('float32')

    n_churn = int(event.sum())
    print(f"Preprocesado completo: {len(X):,} filas × {X.shape[1]} features")
    print(f"Churn en la muestra:   {n_churn} ({n_churn / len(X):.1%})")
    print(
        f"Tiempo (ANTIGUEDAD_CLIENTE, en días) — "
        f"mín: {time.min():.0f} d ({time.min()/365:.1f} a), "
        f"mediana: {time.median():.0f} d ({time.median()/365:.1f} a), "
        f"máx: {time.max():.0f} d ({time.max()/365:.1f} a)"
    )
    return X, time, event
