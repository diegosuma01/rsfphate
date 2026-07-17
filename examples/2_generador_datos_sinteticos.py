"""
Generador de datos sintéticos para el TFG de Diego Suárez Marañón.

Genera dos CSVs:
  1. datos_sinteticos_electricidad.csv  — contratos de electricidad con 3 clusters conocidos
  2. datos_sinteticos_gas.csv           — contratos de gas vinculados por CLIENTE_ID

Columnas idénticas a datos_anonimizados.csv para que el mismo preprocesado funcione.
Añade CLUSTER_REAL (verdad de tierra) para validar que RSF-PHATE recupera los grupos.

Clusters diseñados (electricidad):
  Cluster 0 — "Nuevos en riesgo"   (25%):  churn ~20%, antigüedad mediana ~280 días
  Cluster 1 — "Fieles maduros"     (50%):  churn  ~4%, antigüedad mediana ~2800 días
  Cluster 2 — "Riesgo medio"       (25%):  churn ~10%, antigüedad mediana ~1000 días

Uso:
  python generate_synthetic_data.py             # genera 500 000 filas por defecto
  python generate_synthetic_data.py --n 1000000 # genera 1 000 000 de filas
"""

import argparse
import os
import numpy as np
import pandas as pd

# ── Directorio de salida (misma carpeta que datos_anonimizados.csv) ───────────
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..', '..', 'ClusterScripts'))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de distribuciones de supervivencia
# ─────────────────────────────────────────────────────────────────────────────

def weibull_samples(rng, shape, scale, n):
    """
    Genera n tiempos de supervivencia latentes con distribución Weibull.
    shape < 1 -> riesgo decreciente (muchos fallos tempranos)
    shape = 1 -> riesgo constante (exponencial)
    shape > 1 -> riesgo creciente (fallos tardíos)
    """
    return rng.weibull(shape, size=n) * scale


def apply_censoring(rng, t_latent, t_censor_min=100, t_censor_max=7000):
    """
    Aplica censura aleatoria uniforme.
    Tiempo observado = min(t_latent, t_censura).
    Evento = 1 si el fallo ocurrió antes de la censura.
    """
    t_censor = rng.uniform(t_censor_min, t_censor_max, size=len(t_latent))
    time  = np.minimum(t_latent, t_censor).astype(int).clip(1)
    event = (t_latent <= t_censor).astype(int)
    return time, event


# ─────────────────────────────────────────────────────────────────────────────
# Generador de contratos de electricidad
# ─────────────────────────────────────────────────────────────────────────────

def make_electricity(n=500_000, seed=42):
    rng = np.random.default_rng(seed)

    # ── Proporciones y asignación de clusters ─────────────────────────────────
    props   = [0.25, 0.50, 0.25]          # cluster 0, 1, 2
    cluster = rng.choice([0, 1, 2], size=n, p=props)
    n0, n1, n2 = (cluster == 0).sum(), (cluster == 1).sum(), (cluster == 2).sum()

    print(f"Electricidad: {n:,} filas  (C0={n0:,}  C1={n1:,}  C2={n2:,})")

    # ── Tiempos de supervivencia por cluster ──────────────────────────────────
    # Cluster 0: Weibull con riesgo decreciente -> muchos fallos en los primeros días
    # Cluster 1: Weibull con riesgo creciente  -> fallos muy tardíos, mayoría censurados
    # Cluster 2: Weibull intermedia             -> riesgo moderado
    t_latent = np.empty(n)
    # Escalas calibradas para obtener tasas de churn ~20%, ~4%, ~10%
    # con censura uniforme(85, 6500) — ventana de observacion real del dataset
    t_latent[cluster == 0] = weibull_samples(rng, shape=0.75, scale=24000, n=n0)
    t_latent[cluster == 1] = weibull_samples(rng, shape=2.50, scale=11800, n=n1)
    t_latent[cluster == 2] = weibull_samples(rng, shape=1.20, scale=21500, n=n2)

    time, event = apply_censoring(rng, t_latent, t_censor_min=85, t_censor_max=6500)

    # ── Número de contratos activos del cliente ───────────────────────────────
    # Cluster 1 (fieles) tiende a tener más productos contratados
    n_ctos_base = rng.choice([1,2,3,4,5,6], size=n,
                              p=[0.10, 0.25, 0.30, 0.20, 0.10, 0.05])
    n_ctos_bonus = np.where(cluster == 1,
                            rng.integers(0, 3, size=n), 0)
    n_ctos = (n_ctos_base + n_ctos_bonus).clip(1, 8)

    # ── Cartera media (financiero) ────────────────────────────────────────────
    # C1 (fieles) tiene cartera mucho más alta
    cartera_loc   = np.where(cluster == 0, 5.0,
                    np.where(cluster == 1, 7.2, 5.8))
    cartera_sigma = 0.8
    cartera = np.exp(rng.normal(cartera_loc, cartera_sigma)).round(2).clip(1, 20000)

    # ── Importe de consumo activa ─────────────────────────────────────────────
    consumo_loc = np.where(cluster == 0, 7.5,
                  np.where(cluster == 1, 6.9, 7.2))
    consumo = np.exp(rng.normal(consumo_loc, 0.6)).round(2).clip(50, 30000)

    # ── Deuda vencida (cluster 0 tiene más deuda impagada) ───────────────────
    tiene_deuda = rng.random(n) < np.where(cluster == 0, 0.12,
                                  np.where(cluster == 1, 0.04, 0.07))
    deuda = np.where(tiene_deuda,
                     np.exp(rng.normal(3.5, 1.0, n)).round(2), 0.0)

    # ── Precio máximo ─────────────────────────────────────────────────────────
    precio_max = np.exp(rng.normal(5.5, 0.5, n)).round(2).clip(10, 1000)

    # ── Facturación alta ──────────────────────────────────────────────────────
    fact_alta = np.exp(rng.normal(5.0, 0.8, n)).round(2).clip(1, 5000)
    # ~5% de NaN en facturación
    fact_alta = np.where(rng.random(n) < 0.05, np.nan, fact_alta)

    # ── Días fin de promoción (clusters 0 y 2 tienen más promo activa) ────────
    prob_promo = np.where(cluster == 0, 0.70,
                 np.where(cluster == 1, 0.05, 0.35))
    tiene_promo = rng.random(n) < prob_promo
    dias_promo_max = np.where(cluster == 0, 1403,
                     np.where(cluster == 1,  200, 800))
    dias_fin_promo = np.where(
        tiene_promo,
        rng.integers(30, dias_promo_max + 1, n),
        np.nan
    ).astype(float)

    # ── Días para renovar (clusters 0 y 2 tienen más días pendientes) ─────────
    tiene_renovar = rng.random(n) < 0.60
    dias_renovar = np.where(
        tiene_renovar,
        rng.integers(30, 365, n),
        np.nan
    ).astype(float)
    dias_renovar_mod = np.where(np.isnan(dias_renovar), 0, dias_renovar)

    # ── Días última factura ───────────────────────────────────────────────────
    dias_ult_fact = rng.integers(0, 90, n).astype(float)
    dias_ult_fact = np.where(rng.random(n) < 0.02, np.nan, dias_ult_fact)

    # ── E-factura (% más alto en fieles) ─────────────────────────────────────
    prob_efact = np.where(cluster == 1, 0.92, 0.75)
    efactura = (rng.random(n) < prob_efact).astype(int)

    # ── Descriptor TV ────────────────────────────────────────────────────────
    desc_tv = rng.integers(0, 9, n)

    # ── Contactos, reclamaciones y servicios (comportamiento) ─────────────────
    # Cluster 0: más contactos y reclamaciones (clientes problemáticos)
    lambda_contacto = np.where(cluster == 0, 3.5,
                      np.where(cluster == 1, 1.5, 2.5))
    n_contacto  = rng.poisson(lambda_contacto).clip(0, 30).astype(float)
    n_atc       = rng.poisson(lambda_contacto * 0.3).clip(0, 15).astype(float)
    n_reclam    = rng.poisson(np.where(cluster == 0, 0.8, 0.2)).clip(0, 10).astype(float)
    n_gestion   = rng.poisson(lambda_contacto * 0.5).clip(0, 20).astype(float)
    n_facilita  = rng.poisson(0.3, size=n).clip(0, 5).astype(float)
    n_fact_cobr = rng.poisson(lambda_contacto * 0.8).clip(0, 25).astype(float)
    n_pedido    = rng.poisson(0.2, size=n).clip(0, 5).astype(float)
    n_ventas    = rng.poisson(0.4, size=n).clip(0, 8).astype(float)
    n_ooss_a    = rng.poisson(0.2, size=n).clip(0, 5).astype(float)
    n_ooss_m    = rng.poisson(0.15, size=n).clip(0, 5).astype(float)
    n_ooss_s    = rng.poisson(0.1, size=n).clip(0, 4).astype(float)
    n_ordenes   = (n_ooss_a + n_ooss_m + n_ooss_s).clip(0, 15).astype(float)

    # ── Accesos ML y canje de puntos ─────────────────────────────────────────
    accesos   = (rng.random(n) < 0.60).astype(int)
    canje     = (rng.random(n) < 0.02).astype(int)

    # ── Número de bajas en 36 meses ──────────────────────────────────────────
    n_bajas = rng.poisson(np.where(cluster == 0, 1.2, 0.3)).clip(0, 10).astype(float)

    # ── Contratos por tipo de producto ───────────────────────────────────────
    # NUM_CTOS_L1 -> contratos de luz (casi todos tienen al menos 1)
    ctos_l1 = rng.choice([1, 2, 3, 4], size=n, p=[0.70, 0.20, 0.07, 0.03])
    # NUM_CTOS_L2 -> posiblemente gas (más probable en cluster 1)
    prob_l2 = np.where(cluster == 1, 0.60,
              np.where(cluster == 2, 0.45, 0.30))
    ctos_l2 = np.where(rng.random(n) < prob_l2,
                       rng.choice([1, 2, 3], size=n, p=[0.80, 0.15, 0.05]), 0)
    # NUM_CTOS_02 -> tipo 02 (posiblemente otro producto)
    ctos_02 = np.where(rng.random(n) < 0.22,
                       rng.choice([1, 2], size=n, p=[0.92, 0.08]), 0)
    # NUM_CTOS_05 -> tipo 05
    ctos_05 = np.where(rng.random(n) < 0.55,
                       rng.choice([1, 2, 3], size=n, p=[0.80, 0.15, 0.05]), 0)
    # NUM_CTOS_01 -> tipo 01
    ctos_01 = np.where(rng.random(n) < 0.09,
                       rng.choice([1, 2], size=n, p=[0.90, 0.10]), 0)
    ctos_baja = (rng.random(n) < 0.005).astype(int)
    n_altas_l1 = rng.choice([1, 2, 3, 4, 5], size=n,
                              p=[0.55, 0.25, 0.12, 0.05, 0.03])

    # ── Variables categóricas ─────────────────────────────────────────────────
    # Canal de captación: C1 (fieles) viene de canales propios (CANAL 1-2)
    canal_probs = {
        0: [0.05, 0.10, 0.25, 0.30, 0.20, 0.10],  # canales agresivos
        1: [0.45, 0.30, 0.12, 0.08, 0.03, 0.02],  # canales propios
        2: [0.20, 0.20, 0.20, 0.20, 0.12, 0.08],  # mixto
    }
    canal_vals = np.empty(n, dtype=object)
    for c, probs in canal_probs.items():
        mask = cluster == c
        canal_vals[mask] = [f'CANAL {i+1}' for i in
                            rng.choice(6, size=mask.sum(), p=probs)]

    subcanal_vals = np.array([f'SUBCANAL {i+1}'
                               for i in rng.integers(0, 19, n)])

    n_provincias = 52
    provincia_vals = np.array([f'PROVINCIA {i+1}'
                                for i in rng.integers(0, n_provincias, n)])

    # Renovación/captación: C1 tiende a renovación orgánica
    renov_probs = {
        0: [0.75, 0.15, 0.10],  # mayoría captación nueva
        1: [0.05, 0.85, 0.10],  # mayoría renovación
        2: [0.40, 0.40, 0.20],  # mixto
    }
    renov_labels = ['CAPTACION', 'RENOVACION', 'PORTABILIDAD']
    renov_vals = np.empty(n, dtype=object)
    for c, probs in renov_probs.items():
        mask = cluster == c
        renov_vals[mask] = [renov_labels[i]
                             for i in rng.choice(3, size=mask.sum(), p=probs)]

    gen_vals     = rng.choice(['M', 'F', 'V'], size=n, p=[0.45, 0.40, 0.15])
    alta_adva    = rng.choice(['0', '1'], size=n, p=[0.80, 0.20])

    # ── CLIENTE_ID (para vincular con el dataset de gas) ──────────────────────
    # Un cliente puede tener varios contratos de luz
    n_clientes = int(n / 1.31)  # media ~1.31 contratos luz por cliente
    cliente_id = rng.integers(1, n_clientes + 1, size=n)

    # ── Ensamblado del DataFrame ──────────────────────────────────────────────
    df = pd.DataFrame({
        'NUM_CONTRATO':               np.arange(1, n + 1),
        'CLIENTE_ID':                 cliente_id,
        'CLUSTER_REAL':               cluster,         # verdad de tierra (no en datos reales)
        'ANTIGUEDAD_CLIENTE':         time,
        'TARGET':                     event,
        'CANAL_INPUT':                canal_vals,
        'SUBCANAL_INPUT':             subcanal_vals,
        'PROVINCIA_PS':               provincia_vals,
        'RENOVACION_CAPTACION_MOD':   renov_vals,
        'GEN_MOD':                    gen_vals,
        'IND_Alta_Adva':              alta_adva,
        'CARTERA_MEDIA':              cartera,
        'IMP_CONSUMO_ACTIVA_MOD':     consumo,
        'IMP_DEUDA_VENCIDA':          deuda,
        'IMP_PRECIO_MAX':             precio_max,
        'FACTURACION_ALTA':           fact_alta,
        'DIAS_FIN_PROMO':             dias_fin_promo,
        'DIAS_PARA_RENOVAR_MOD':      dias_renovar_mod,
        'DIAS_ULTIMA_FACTURA':        dias_ult_fact,
        'EFACTURA':                   efactura,
        'DESC_TV_MOD':                desc_tv,
        'N_CONTACTO_FECHA_REF_3':     n_contacto,
        'N_ATC_FECHA_REF_3_MOD':      n_atc,
        'N_RECLAM_FECHA_REF_3_MOD':   n_reclam,
        'N_GESTION_FECHA_REF_3_MOD':  n_gestion,
        'N_FACILITA_FECHA_REF_3_MOD': n_facilita,
        'N_FACT_COBR_FECHA_REF_3_MOD':n_fact_cobr,
        'N_PEDIDOINFO_FECHA_REF_3_MOD':n_pedido,
        'N_VENTAS_CONT_FECHA_REF_3_MOD':n_ventas,
        'N_OOSS_ASIST_MOD':           n_ooss_a,
        'N_OOSS_MANT_MOD':            n_ooss_m,
        'N_OOSS_SSAA_MOD':            n_ooss_s,
        'N_ORDENES_SERVICIO_MOD':     n_ordenes,
        'ACCESOS_AC_ML_3MESES_MOD':   accesos,
        'CANJE_PUNTOS_FECHA_REF_3_BIN':canje,
        'N_CTOS_ACTIVOS_IC':          n_ctos,
        'N_CTOS_BAJA_UMES_IC_MOD':    ctos_baja,
        'NUM_CTOS_01_MOD':            ctos_01,
        'NUM_CTOS_02_MOD':            ctos_02,
        'NUM_CTOS_05_MOD':            ctos_05,
        'NUM_CTOS_L1_MOD':            ctos_l1,
        'NUM_CTOS_L2_MOD':            ctos_l2,
        'NBAJAS_36M_MOD':             n_bajas,
        'N_ALTAS_L1_IC':              n_altas_l1,
    })

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Generador de contratos de gas
# ─────────────────────────────────────────────────────────────────────────────

def make_gas(df_elec, seed=43):
    """
    Genera contratos de gas vinculados con el dataset de electricidad.
    Solo el ~50% de los clientes de luz tienen también gas.
    La supervivencia del gas está correlacionada con la del cluster de luz:
      - Clientes del cluster 1 (fieles en luz) -> también fieles en gas
      - Clientes del cluster 0 (riesgo en luz) -> mayor riesgo en gas también
    """
    rng = np.random.default_rng(seed)

    # Selección de clientes con contrato de gas
    clientes_unicos = df_elec['CLIENTE_ID'].unique()
    n_clientes = len(clientes_unicos)
    tiene_gas = rng.random(n_clientes) < 0.50
    clientes_gas = clientes_unicos[tiene_gas]

    # Tomamos un contrato por cliente (el primero que aparece)
    df_rep = (df_elec.drop_duplicates('CLIENTE_ID')
                     .set_index('CLIENTE_ID')
                     .loc[clientes_gas]
                     .reset_index())

    n = len(df_rep)
    cluster = df_rep['CLUSTER_REAL'].values
    print(f"Gas:          {n:,} filas  (50% de {n_clientes:,} clientes)")

    # ── Tiempos de supervivencia del contrato de gas ──────────────────────────
    # Correlacionados con el cluster de luz pero con algo de ruido propio
    t_latent = np.empty(n)
    t_latent[cluster == 0] = weibull_samples(rng, shape=0.80, scale=22000, n=(cluster==0).sum())
    t_latent[cluster == 1] = weibull_samples(rng, shape=2.20, scale=12000, n=(cluster==1).sum())
    t_latent[cluster == 2] = weibull_samples(rng, shape=1.10, scale=20000, n=(cluster==2).sum())

    time_gas, event_gas = apply_censoring(rng, t_latent, t_censor_min=85, t_censor_max=6500)

    # ── Features del contrato de gas ─────────────────────────────────────────
    # Cartera de gas: menor que electricidad, también correlacionada con cluster
    cartera_loc_gas = np.where(cluster == 0, 4.5,
                      np.where(cluster == 1, 6.5, 5.2))
    cartera_gas = np.exp(rng.normal(cartera_loc_gas, 0.7)).round(2).clip(1, 8000)

    # Consumo de gas
    consumo_loc_gas = np.where(cluster == 0, 6.8,
                      np.where(cluster == 1, 6.2, 6.5))
    consumo_gas = np.exp(rng.normal(consumo_loc_gas, 0.5)).round(2).clip(20, 15000)

    # Precio del gas
    precio_gas = np.exp(rng.normal(4.8, 0.4, n)).round(2).clip(5, 500)

    # Días fin de promo gas
    prob_promo_gas = np.where(cluster == 0, 0.65,
                    np.where(cluster == 1, 0.08, 0.30))
    tiene_promo_gas = rng.random(n) < prob_promo_gas
    dias_fin_promo_gas = np.where(
        tiene_promo_gas,
        rng.integers(30, 1200, n), np.nan
    ).astype(float)

    # Días para renovar gas
    dias_renovar_gas = np.where(
        rng.random(n) < 0.55, rng.integers(30, 365, n), 0
    ).astype(float)

    # Contactos y reclamaciones gas
    lambda_cont_gas = np.where(cluster == 0, 2.5, np.where(cluster == 1, 1.0, 1.8))
    n_contacto_gas  = rng.poisson(lambda_cont_gas).clip(0, 20).astype(float)
    n_reclam_gas    = rng.poisson(np.where(cluster == 0, 0.6, 0.15)).clip(0, 8).astype(float)
    n_ooss_gas      = rng.poisson(0.3, size=n).clip(0, 6).astype(float)

    # Canal gas (puede diferir del canal de luz)
    canal_gas = np.array([f'CANAL {i+1}' for i in rng.integers(0, 6, n)])
    subcanal_gas = np.array([f'SUBCANAL {i+1}' for i in rng.integers(0, 19, n)])
    efactura_gas = (rng.random(n) < np.where(cluster == 1, 0.90, 0.72)).astype(int)

    # Deuda gas
    tiene_deuda_gas = rng.random(n) < np.where(cluster == 0, 0.10,
                                   np.where(cluster == 1, 0.03, 0.06))
    deuda_gas = np.where(tiene_deuda_gas, np.exp(rng.normal(3.2, 0.9, n)).round(2), 0.0)

    # ── Ensamblado ────────────────────────────────────────────────────────────
    df_gas = pd.DataFrame({
        'NUM_CONTRATO_GAS':       np.arange(1, n + 1),
        'CLIENTE_ID':             df_rep['CLIENTE_ID'].values,
        'CLUSTER_REAL':           cluster,
        'ANTIGUEDAD_GAS':         time_gas,
        'TARGET_GAS':             event_gas,
        'CANAL_INPUT':            canal_gas,
        'SUBCANAL_INPUT':         subcanal_gas,
        'PROVINCIA_PS':           df_rep['PROVINCIA_PS'].values,   # misma provincia
        'CARTERA_MEDIA_GAS':      cartera_gas,
        'IMP_CONSUMO_GAS':        consumo_gas,
        'IMP_PRECIO_GAS':         precio_gas,
        'IMP_DEUDA_VENCIDA_GAS':  deuda_gas,
        'DIAS_FIN_PROMO_GAS':     dias_fin_promo_gas,
        'DIAS_PARA_RENOVAR_GAS':  dias_renovar_gas,
        'EFACTURA_GAS':           efactura_gas,
        'N_CONTACTO_GAS':         n_contacto_gas,
        'N_RECLAM_GAS':           n_reclam_gas,
        'N_OOSS_GAS':             n_ooss_gas,
        # Referencia cruzada con el cluster de electricidad
        'CLUSTER_REAL_LUZ':       cluster,
        'RENOVACION_CAPTACION_MOD': df_rep['RENOVACION_CAPTACION_MOD'].values,
        'GEN_MOD':                df_rep['GEN_MOD'].values,
    })

    return df_gas


# ─────────────────────────────────────────────────────────────────────────────
# Diagnóstico rápido tras la generación
# ─────────────────────────────────────────────────────────────────────────────

def diagnostico(df_elec, df_gas):
    print('\n' + '='*60)
    print('DIAGNÓSTICO — ELECTRICIDAD')
    print('='*60)
    print(f'Shape: {df_elec.shape}')
    print(f'Churn rate global: {df_elec.TARGET.mean():.2%}')
    print()
    resumen = df_elec.groupby('CLUSTER_REAL').agg(
        n=('TARGET','count'),
        churn_pct=('TARGET','mean'),
        mediana_dias=('ANTIGUEDAD_CLIENTE', 'median'),
        cartera_media=('CARTERA_MEDIA','mean'),
        pct_con_promo=('DIAS_FIN_PROMO', lambda x: x.notna().mean()),
    ).round(3)
    resumen['churn_pct'] = (resumen['churn_pct']*100).round(1).astype(str)+'%'
    resumen['pct_con_promo'] = (resumen['pct_con_promo']*100).round(0).astype(int).astype(str)+'%'
    print(resumen.to_string())

    print('\n' + '='*60)
    print('DIAGNÓSTICO — GAS')
    print('='*60)
    print(f'Shape: {df_gas.shape}')
    print(f'Churn rate global: {df_gas.TARGET_GAS.mean():.2%}')
    print()
    resumen_gas = df_gas.groupby('CLUSTER_REAL').agg(
        n=('TARGET_GAS','count'),
        churn_pct=('TARGET_GAS','mean'),
        mediana_dias=('ANTIGUEDAD_GAS','median'),
    ).round(3)
    resumen_gas['churn_pct'] = (resumen_gas['churn_pct']*100).round(1).astype(str)+'%'
    print(resumen_gas.to_string())

    print('\n' + '='*60)
    print('FICHEROS GENERADOS')
    print('='*60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Genera datos sintéticos de churn eléctrico')
    parser.add_argument('--n', type=int, default=500_000,
                        help='Número de contratos de electricidad (default: 500000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria (default: 42)')
    parser.add_argument('--out', type=str, default=OUT_DIR,
                        help='Directorio de salida')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f'Generando {args.n:,} contratos de electricidad (seed={args.seed})...')
    df_elec = make_electricity(n=args.n, seed=args.seed)

    print('Generando contratos de gas vinculados...')
    df_gas  = make_gas(df_elec, seed=args.seed + 1)

    diagnostico(df_elec, df_gas)

    path_elec = os.path.join(args.out, 'datos_sinteticos_electricidad.csv')
    path_gas  = os.path.join(args.out, 'datos_sinteticos_gas.csv')

    print(f'Guardando electricidad -> {path_elec}')
    df_elec.to_csv(path_elec, index=False, encoding='utf-8')

    print(f'Guardando gas          -> {path_gas}')
    df_gas.to_csv(path_gas, index=False, encoding='utf-8')

    size_elec = os.path.getsize(path_elec) / 1_000_000
    size_gas  = os.path.getsize(path_gas)  / 1_000_000
    print(f'\nTamaño electricidad: {size_elec:.0f} MB')
    print(f'Tamaño gas:          {size_gas:.0f} MB')
    print('\nListo.')


if __name__ == '__main__':
    main()
