
# TFM — Detección Temprana de Sepsis con Deep Learning

Trabajo Fin de Máster. Modelo LSTM para predicción de sepsis en UCI usando datos de **MIMIC-IV**.

---

## Objetivo

Predecir el inicio de sepsis en pacientes de UCI a partir de los signos vitales y analíticas de las **48 horas previas** al onset, usando series temporales con intervalos de 1 hora.

---

## Dataset — MIMIC-IV

Base de datos clínica anonimizada del Beth Israel Deaconess Medical Center (Boston).

| Tabla | Registros | Descripción |
|-------|-----------|-------------|
| `hosp/admissions` | 546,028 | Admisiones hospitalarias |
| `icu/icustays` | 94,458 | Estancias en UCI |
| `icu/chartevents` | ~433M | Signos vitales minuto a minuto |
| `hosp/labevents` | ~158M | Analíticas de laboratorio |
| `icu/d_items` | 4,095 | Diccionario de variables UCI |
| `hosp/d_labitems` | 1,650 | Diccionario de variables de laboratorio |

---

## Cohorte de Sepsis

| Método | Admisiones |
|--------|-----------|
| Códigos ICD-9 explícitos | 8,705 |
| Códigos ICD-10 | 13,763 |
| Algoritmo Angus (ICD-9) | 63,066 |
| **Total admisiones** | **76,828** |
| **Estancias UCI (cohorte final)** | **29,966** |

### Identificación de sepsis

- **ICD-9**: Códigos `995.91`, `995.92`, `785.52`, `038.x` (septicemia) + Algoritmo Angus (infección + disfunción orgánica)
- **ICD-10**: Prefijos `A40.x`, `A41.x` + códigos adicionales (`R65.20`, `R65.21`, `B37.7`, etc.)

### Claves de enlace entre tablas

| Campo | Descripción |
|-------|-------------|
| `subject_id` | Identificador único de paciente |
| `hadm_id` | Identificador único de admisión hospitalaria |
| `stay_id` | Identificador único de estancia UCI |

---

## Extracción de Features

### Ventana temporal
- **Referencia**: `intime` de la estancia UCI (proxy del onset de sepsis)
- **Ventana**: 48h previas al onset
- **Granularidad**: intervalos de 1 hora (`hour_bucket`)

### Variables SOFA — Signos vitales (`chartevents` + `d_items`)

| itemid | Variable |
|--------|----------|
| 220045 | Heart Rate |
| 220179 | Non Invasive Blood Pressure systolic |
| 220180 | Non Invasive Blood Pressure diastolic |
| 220052 | Arterial Blood Pressure mean |
| 220210 | Respiratory Rate |
| 223762 | Temperature Celsius |
| 220277 | SpO2 / Pulse Oximetry |
| 229867 | FiO2 |
| 220339 / 223900 / 223901 | GCS (Eye, Verbal, Motor) |

### Variables SOFA — Laboratorio (`labevents` + `d_labitems`)

| itemid | Variable |
|--------|----------|
| 51265 | Platelet Count |
| 50912 | Creatinine |
| 50885 | Bilirubin Total |
| 50821 | PaO2 |
| 50820 | pH |
| 50813 | Lactate |
| 51006 | Blood Urea Nitrogen |

### Estructura de datos final

```
features_df: [stay_id × hour_bucket × N_features]
  - Rango: hour_bucket ∈ [-48, 0)
  - Agregación: media por hora cuando hay múltiples mediciones
```

---

## Arquitectura del Proyecto

```
TFM_DeepLearning/
├── data/
│   ├── hosp/          # Datos hospitalarios (MIMIC-IV hosp module)
│   └── icu/           # Datos UCI (MIMIC-IV icu module)
├── notebooks/
│   └── 01_data_analysis.ipynb   # Análisis y extracción de datos
├── src/               # Código fuente (pipelines, modelos)
├── models/            # Modelos entrenados
├── outputs/           # Resultados y métricas
└── results/           # Visualizaciones
```

---

## TODO

### 1 — Data Analysis & Extraction
- [ ] Convertir todo el proceso del notebook 1 a una Pipeline
- [ ] Analizar outliers, NaNs y estrategia de imputación
- [ ] Evaluar uso de `sepsis3` (tabla derivada MIMIC-IV) como onset más preciso que `intime`
- [ ] Comparar ventanas de 24h vs 48h

### 2 — Modelo LSTM
- [ ] Diseño de arquitectura LSTM
- [ ] Definir estrategia de padding/masking para series de longitud variable
- [ ] Validación cruzada por paciente (evitar data leakage)
- [ ] Métricas: AUROC, AUPRC, sensibilidad, especificidad

### 3 — Evaluación
- [ ] Comparativa con baseline (regresión logística, XGBoost)
- [ ] Análisis de importancia de features (SHAP)
