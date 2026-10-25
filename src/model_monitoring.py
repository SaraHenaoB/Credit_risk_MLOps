#From src/model_monitoring.py
# run this command: streamlit run model_monitoring.py
# run this command: streamlit run src/model_monitoring.py
#librerías

import numpy as np
import pandas as pd
import streamlit as st                                          #Librería para crear la aplicación  en Streamlit
import matplotlib.pyplot as plt                                 #librerías de visualización

from scipy.stats import ks_2samp, chi2_contingency              #Kolmovorov-Smirnov variables numericas, chi chuadrado variable categoricas
from scipy.spatial.distance import jensenshannon

from cargar_datos import cargarDatos                            #importar el método cargar datos


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Monitoreo de Data Drift",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard de Monitoreo de Data Drift")
st.markdown(
    """
    Este dashboard compara la distribución de los datos de referencia
    con la de los datos más recientes para identificar posibles cambios
    en las variables de entrada del modelo.
    """
)

TARGET = "Pago_atiempo"
DATE_COLUMN = "fecha_prestamo"

# Umbrales de monitoreo: son parámetros iniciales configurables.
PSI_THRESHOLD = 0.20
JS_THRESHOLD = 0.10
P_VALUE_THRESHOLD = 0.05
DRIFT_ALERT_THRESHOLD = 2

# Columnas que no deben usarse como variables predictoras.
LEAKAGE_FEATURES = [
    "saldo_mora",
    "saldo_mora_codeudor",
    "saldo_total",
    "saldo_principal",
    "puntaje"
]


# ============================================================
# 2. CARGA Y DIVISIÓN TEMPORAL DE LOS DATOS
# ============================================================

@st.cache_data(ttl=300)
def load_monitoring_data():
    """
    Carga los datos, ordena por fecha y divide cronológicamente:
    80 % inicial como referencia y 20 % final como datos recientes.
    """

    df = cargarDatos()

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(
            "La función cargarDatos() no devolvió un DataFrame válido "
            "o el DataFrame está vacío."
        )

    df = df.copy()

    if TARGET not in df.columns:
        raise ValueError(
            f"No se encontró la columna objetivo '{TARGET}' en los datos."
        )

    # Convertir la fecha sin imponer un formato fijo.
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(
            df[DATE_COLUMN],
            errors="coerce"
        )

        df = df.dropna(subset=[DATE_COLUMN])
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(
            "No hay suficientes registros válidos para comparar dos periodos."
        )

    # Excluir columnas con riesgo de fuga de información.
    df = df.drop(
        columns=[
            col for col in LEAKAGE_FEATURES
            if col in df.columns
        ],
        errors="ignore"
    )

    # Guardar las fechas y todas las variables para el gráfico temporal.
    # La fecha y el objetivo no se incluyen como variables de entrada.
    if DATE_COLUMN in df.columns:
        dates_all = df[DATE_COLUMN].copy().reset_index(drop=True)
    else:
        dates_all = pd.Series(pd.NaT, index=range(len(df)))

    excluded_columns = [
        col for col in [TARGET, DATE_COLUMN]
        if col in df.columns
    ]

    X = df.drop(columns=excluded_columns, errors="ignore")

    if X.empty:
        raise ValueError(
            "No quedaron variables para monitorear después de excluir "
            "el objetivo, la fecha y las columnas de fuga."
        )

    # División cronológica estricta: no se mezclan los registros.
    split_index = int(len(X) * 0.80)
    split_index = max(1, min(split_index, len(X) - 1))

    X_all = X.copy().reset_index(drop=True)
    X_ref = X.iloc[:split_index].copy()
    X_new = X.iloc[split_index:].copy()

    return X_ref, X_new, X_all, dates_all, len(df), split_index


# ============================================================
# 3. MÉTRICAS DE DATA DRIFT
# ============================================================

def calculate_psi(reference, current, bins=10):
    """
    Calcula Population Stability Index (PSI).

    Para variables numéricas utiliza intervalos basados en los
    cuantiles de la referencia. Para variables categóricas compara
    las proporciones de cada categoría.
    """

    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()

    if ref.empty or cur.empty:
        return np.nan

    is_numeric = (
        pd.api.types.is_numeric_dtype(ref)
        and pd.api.types.is_numeric_dtype(cur)
    )

    if is_numeric:
        # Los límites se calculan solamente con los datos de referencia.
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(ref.quantile(quantiles).to_numpy())

        if len(edges) < 2:
            # Si la referencia es constante, se compara como categoría.
            ref_values = ref.astype(str)
            cur_values = cur.astype(str)
            categories = sorted(
                set(ref_values.unique()) | set(cur_values.unique())
            )

            ref_counts = (
                ref_values.value_counts()
                .reindex(categories, fill_value=0)
                .to_numpy(dtype=float)
            )
            cur_counts = (
                cur_values.value_counts()
                .reindex(categories, fill_value=0)
                .to_numpy(dtype=float)
            )
        else:
            # Extender los extremos para incluir todos los valores nuevos.
            edges[0] = -np.inf
            edges[-1] = np.inf

            ref_bins = pd.cut(
                ref, bins=edges, include_lowest=True
            )
            cur_bins = pd.cut(
                cur, bins=edges, include_lowest=True
            )

            ref_counts = (
                ref_bins.value_counts(sort=False)
                .to_numpy(dtype=float)
            )
            cur_counts = (
                cur_bins.value_counts(sort=False)
                .to_numpy(dtype=float)
            )
    else:
        ref_values = ref.astype(str)
        cur_values = cur.astype(str)

        categories = sorted(
            set(ref_values.unique()) | set(cur_values.unique())
        )

        ref_counts = (
            ref_values.value_counts()
            .reindex(categories, fill_value=0)
            .to_numpy(dtype=float)
        )
        cur_counts = (
            cur_values.value_counts()
            .reindex(categories, fill_value=0)
            .to_numpy(dtype=float)
        )

    # Suavizado para evitar divisiones o logaritmos de cero.
    epsilon = 1e-6

    ref_pct = (ref_counts + epsilon) / (
        ref_counts.sum() + epsilon * len(ref_counts)
    )
    cur_pct = (cur_counts + epsilon) / (
        cur_counts.sum() + epsilon * len(cur_counts)
    )

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def calculate_js(reference, current, bins=10):
    """
    Calcula la divergencia Jensen-Shannon entre dos distribuciones.
    Su valor está entre 0 y 1 cuando se usa la base logarítmica 2.
    """

    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()

    if ref.empty or cur.empty:
        return np.nan

    is_numeric = (
        pd.api.types.is_numeric_dtype(ref)
        and pd.api.types.is_numeric_dtype(cur)
    )

    if is_numeric:
        # Se usan los mismos intervalos para ambas distribuciones.
        edges = np.unique(
            ref.quantile(np.linspace(0, 1, bins + 1)).to_numpy()
        )

        if len(edges) < 2:
            ref_values = ref.astype(str)
            cur_values = cur.astype(str)
            categories = sorted(
                set(ref_values.unique()) | set(cur_values.unique())
            )

            ref_counts = (
                ref_values.value_counts()
                .reindex(categories, fill_value=0)
                .to_numpy(dtype=float)
            )
            cur_counts = (
                cur_values.value_counts()
                .reindex(categories, fill_value=0)
                .to_numpy(dtype=float)
            )
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf

            ref_bins = pd.cut(
                ref, bins=edges, include_lowest=True
            )
            cur_bins = pd.cut(
                cur, bins=edges, include_lowest=True
            )

            ref_counts = (
                ref_bins.value_counts(sort=False)
                .to_numpy(dtype=float)
            )
            cur_counts = (
                cur_bins.value_counts(sort=False)
                .to_numpy(dtype=float)
            )
    else:
        ref_values = ref.astype(str)
        cur_values = cur.astype(str)

        categories = sorted(
            set(ref_values.unique()) | set(cur_values.unique())
        )

        ref_counts = (
            ref_values.value_counts()
            .reindex(categories, fill_value=0)
            .to_numpy(dtype=float)
        )
        cur_counts = (
            cur_values.value_counts()
            .reindex(categories, fill_value=0)
            .to_numpy(dtype=float)
        )

    epsilon = 1e-6

    ref_pct = (ref_counts + epsilon) / (
        ref_counts.sum() + epsilon * len(ref_counts)
    )
    cur_pct = (cur_counts + epsilon) / (
        cur_counts.sum() + epsilon * len(cur_counts)
    )

    return float(jensenshannon(ref_pct, cur_pct, base=2))


def calculate_data_drift(X_ref, X_new):
    """
    Calcula métricas para cada variable y determina si hay drift.

    Numéricas:
        - Kolmogorov-Smirnov (KS)
        - PSI
        - Jensen-Shannon (JS)

    Categóricas:
        - Chi-cuadrado
        - PSI
        - Jensen-Shannon (JS)
    """

    results = []

    # Incluye enteros, decimales y otros tipos numéricos.
    numerical_features = X_ref.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Variables categóricas, texto y booleanas.
    categorical_features = X_ref.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    for feature in numerical_features:
        ref = pd.to_numeric(
            X_ref[feature], errors="coerce"
        ).dropna()
        cur = pd.to_numeric(
            X_new[feature], errors="coerce"
        ).dropna()

        if ref.empty or cur.empty:
            continue

        # KS requiere datos suficientes y variabilidad en las muestras.
        if ref.nunique() > 1 or cur.nunique() > 1:
            ks_stat, p_value = ks_2samp(ref, cur)
        else:
            ks_stat, p_value = 0.0, 1.0

        psi = calculate_psi(ref, cur)
        js = calculate_js(ref, cur)

        drift_detected = bool(
            p_value < P_VALUE_THRESHOLD
            or (not np.isnan(psi) and psi >= PSI_THRESHOLD)
            or (not np.isnan(js) and js >= JS_THRESHOLD)
        )

        results.append({
            "Variable": feature,
            "Tipo": "Numérica",
            "Métrica": "KS",
            "Estadístico": float(ks_stat),
            "P_Value": float(p_value),
            "PSI": psi,
            "Jensen_Shannon": js,
            "Drift_Detected": drift_detected
        })

    for feature in categorical_features:
        ref = X_ref[feature].dropna().astype(str)
        cur = X_new[feature].dropna().astype(str)

        if ref.empty or cur.empty:
            continue

        categories = sorted(
            set(ref.unique()) | set(cur.unique())
        )

        ref_counts = (
            ref.value_counts()
            .reindex(categories, fill_value=0)
        )
        cur_counts = (
            cur.value_counts()
            .reindex(categories, fill_value=0)
        )

        contingency_table = np.vstack([
            ref_counts.to_numpy(),
            cur_counts.to_numpy()
        ])

        # Chi-cuadrado necesita una tabla con más de una categoría
        # y observaciones suficientes para calcular la prueba.
        if len(categories) > 1:
            try:
                chi2_stat, p_value, _, _ = chi2_contingency(
                    contingency_table
                )
            except ValueError:
                chi2_stat, p_value = np.nan, np.nan
        else:
            chi2_stat, p_value = 0.0, 1.0

        psi = calculate_psi(ref, cur)
        js = calculate_js(ref, cur)

        p_value_drift = (
            not np.isnan(p_value)
            and p_value < P_VALUE_THRESHOLD
        )

        drift_detected = bool(
            p_value_drift
            or (not np.isnan(psi) and psi >= PSI_THRESHOLD)
            or (not np.isnan(js) and js >= JS_THRESHOLD)
        )

        results.append({
            "Variable": feature,
            "Tipo": "Categórica",
            "Métrica": "Chi-cuadrado",
            "Estadístico": chi2_stat,
            "P_Value": p_value,
            "PSI": psi,
            "Jensen_Shannon": js,
            "Drift_Detected": drift_detected
        })

    return pd.DataFrame(results)


# ============================================================
# 4. ESTILO DE LA TABLA
# ============================================================

def highlight_drift(row):
    """
    Colorea la fila según la columna Estado que se muestra
    en la tabla. No depende de Drift_Detected, porque esa
    columna se elimina de la tabla de presentación.
    """

    estado = str(row["Estado"])

    if estado.startswith("🔴"):
        color = (
            "background-color: #FCE4E4; "
            "color: #8B0000; font-weight: bold"
        )
    else:
        color = (
            "background-color: #E5F4E3; "
            "color: #205522"
        )

    return [color] * len(row)


# ============================================================
# 5. INTERFAZ DEL DASHBOARD
# ============================================================

with st.sidebar:
    st.header("Configuración")

    simulate_drift = st.checkbox(
        "Simular drift para demostración",
        value=False,
        help=(
            "Modifica temporalmente algunas variables de los datos "
            "recientes para comprobar cómo responde el dashboard. "
            "No modifica el archivo original."
        )
    )

    if st.button("Actualizar monitoreo", width="stretch"):
        load_monitoring_data.clear()
        st.rerun()


try:
    X_ref, X_new, X_all, dates_all, total_rows, split_index = (
        load_monitoring_data()
    )

    # Copia para que la simulación no modifique los datos originales.
    X_current = X_new.copy()

    if simulate_drift:
        simulated_columns = []

        # Solo se modifican variables que realmente existan.
        if "salario_cliente" in X_current.columns:
            X_current["salario_cliente"] = (
                X_current["salario_cliente"] * 0.70
            )
            simulated_columns.append("salario_cliente")

        if "promedio_ingresos_datacredito" in X_current.columns:
            X_current["promedio_ingresos_datacredito"] = (
                X_current["promedio_ingresos_datacredito"] * 0.70
            )
            simulated_columns.append(
                "promedio_ingresos_datacredito"
            )

        if "capital_prestado" in X_current.columns:
            X_current["capital_prestado"] = (
                X_current["capital_prestado"] * 1.40
            )
            simulated_columns.append("capital_prestado")

        if simulated_columns:
            st.warning(
                "Modo de demostración activo. Se ha simulado drift "
                "en: " + ", ".join(simulated_columns) + "."
            )
        else:
            st.info(
                "No se encontraron las columnas configuradas para la "
                "simulación. Se mostrarán los datos sin modificar."
            )

    # Métricas principales.
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Registros analizados", f"{total_rows:,}")
    col2.metric("Datos de referencia", f"{len(X_ref):,}")
    col3.metric("Datos recientes", f"{len(X_current):,}")
    col4.metric("Variables monitoreadas", f"{X_ref.shape[1]:,}")

    st.caption(
        "División cronológica: primeros registros para referencia y "
        "últimos registros para comparación. El objetivo y las columnas "
        "excluidas por fuga de información no se usan en este análisis."
    )

    # Ejecutar el monitoreo.
    drift_results = calculate_data_drift(X_ref, X_current)

    if drift_results.empty:
        st.warning(
            "No se pudieron calcular métricas de drift. Revisa los tipos "
            "de las columnas y si existen datos válidos en ambos periodos."
        )
        st.stop()

    drift_count = int(drift_results["Drift_Detected"].sum())
    total_features = len(drift_results)

    st.subheader("Resultado general")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric(
        "Variables con drift",
        f"{drift_count} de {total_features}"
    )
    metric_col2.metric(
        "Variables sin drift",
        f"{total_features - drift_count} de {total_features}"
    )
    metric_col3.metric(
        "Porcentaje con drift",
        f"{(drift_count / total_features) * 100:.1f}%"
    )

    if drift_count > DRIFT_ALERT_THRESHOLD:
        st.error(
            f"Alerta global: se detectó drift en {drift_count} variables. "
            "El número supera el umbral configurado de "
            f"{DRIFT_ALERT_THRESHOLD} variables."
        )
    elif drift_count > 0:
        st.warning(
            f"Se detectó drift en {drift_count} variable(s). "
            "Conviene revisar las variables señaladas."
        )
    else:
        st.success(
            "No se detectó drift con los umbrales configurados."
        )

    # Tabla de resultados.
    st.subheader("Detalle por variable")

    display_df = drift_results.copy()

    display_df["Estado"] = display_df["Drift_Detected"].map({
        True: "🔴 Drift detectado",
        False: "🟢 Sin drift detectado"
    })

    display_df = display_df.drop(
        columns=["Drift_Detected"]
    )

    # Redondear métricas para facilitar la lectura.
    for column in [
        "Estadístico", "P_Value", "PSI", "Jensen_Shannon"
    ]:
        if column in display_df.columns:
            display_df[column] = display_df[column].round(4)

    # FIX: highlight_drift usa Estado, que sí existe en display_df.
    styled_df = display_df.style.apply(
        highlight_drift,
        axis=1
    )

    # FIX: width="stretch" reemplaza use_container_width=True.
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True
    )

    # ========================================================
    # GUÍA RÁPIDA: INTERPRETACIÓN DE LAS MÉTRICAS
    # ========================================================

    with st.expander(
        "📘 Cómo interpretar las métricas y sus umbrales",
        expanded=True
    ):
        interpretation_df = pd.DataFrame([
            {
                "Métrica": "KS (estadístico)",
                "Qué mide": "Máxima diferencia entre distribuciones numéricas.",
                "Cómo interpretarla": "Entre 0 y 1; cuanto mayor, mayor diferencia.",
                "Umbral de referencia": "Sin umbral fijo en este proyecto."
            },
            {
                "Métrica": "P_Value (KS / Chi²)",
                "Qué mide": "Evidencia estadística de cambio entre periodos.",
                "Cómo interpretarla": "Un valor menor indica mayor evidencia de diferencia.",
                "Umbral de referencia": "< 0.05: posible drift."
            },
            {
                "Métrica": "PSI",
                "Qué mide": "Magnitud del cambio en la distribución.",
                "Cómo interpretarla": "< 0.10: pequeño; 0.10–0.20: moderado; ≥ 0.20: importante.",
                "Umbral de referencia": "≥ 0.20: drift."
            },
            {
                "Métrica": "Jensen_Shannon",
                "Qué mide": "Divergencia entre dos distribuciones.",
                "Cómo interpretarla": "Entre 0 y 1; cerca de 0 significa mayor similitud.",
                "Umbral de referencia": "≥ 0.10: drift."
            },
            {
                "Métrica": "Drift_Detected",
                "Qué mide": "Resultado conjunto de los criterios configurados.",
                "Cómo interpretarla": "True: al menos un criterio detectó drift; False: ninguno lo superó.",
                "Umbral de referencia": "Se activa si KS/Chi² p-valor < 0.05, PSI ≥ 0.20 o JS ≥ 0.10."
            }
        ])

        st.dataframe(
            interpretation_df,
            width="stretch",
            hide_index=True
        )

        st.caption(
            "Nota: los umbrales son criterios iniciales, no límites universales. "
            "Los p-valores dependen del tamaño de la muestra y no miden por sí "
            "solos la magnitud del cambio. La ausencia de drift tampoco garantiza "
            "que el modelo conserve su rendimiento."
        )

    # ========================================================
    # 6. VISUALIZACIÓN DE UNA VARIABLE
    # ========================================================

    st.subheader("Comparación de distribuciones")

    available_features = [
        feature for feature in X_ref.columns
        if feature in X_current.columns
    ]

    selected_feature = st.selectbox(
        "Selecciona una variable para comparar",
        available_features
    )

    if selected_feature:
        reference_values = X_ref[selected_feature].dropna()
        current_values = X_current[selected_feature].dropna()

        fig, ax = plt.subplots(figsize=(10, 4))

        if (
            pd.api.types.is_numeric_dtype(reference_values)
            and pd.api.types.is_numeric_dtype(current_values)
        ):
            ax.hist(
                reference_values,
                bins=25,
                alpha=0.55,
                density=True,
                label="Referencia"
            )
            ax.hist(
                current_values,
                bins=25,
                alpha=0.55,
                density=True,
                label="Datos recientes"
            )
            ax.set_ylabel("Densidad")
        else:
            ref_pct = (
                reference_values.astype(str)
                .value_counts(normalize=True)
            )
            cur_pct = (
                current_values.astype(str)
                .value_counts(normalize=True)
            )

            categories = sorted(
                set(ref_pct.index) | set(cur_pct.index)
            )

            comparison = pd.DataFrame({
                "Referencia": ref_pct.reindex(
                    categories, fill_value=0
                ),
                "Datos recientes": cur_pct.reindex(
                    categories, fill_value=0
                )
            })

            comparison.plot(
                kind="bar",
                ax=ax
            )
            ax.set_ylabel("Proporción")
            ax.tick_params(axis="x", rotation=45)

        ax.set_title(
            f"Distribución de {selected_feature}"
        )
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # Mostrar las métricas calculadas para la variable seleccionada.
        feature_result = drift_results[
            drift_results["Variable"] == selected_feature
        ]

        if not feature_result.empty:
            result = feature_result.iloc[0]

            st.write(
                f"**Tipo:** {result['Tipo']}  \n"
                f"**Prueba estadística:** {result['Métrica']}  \n"
                f"**PSI:** {result['PSI']:.4f}  \n"
                f"**Jensen-Shannon:** "
                f"{result['Jensen_Shannon']:.4f}"
            )

            if result["Drift_Detected"]:
                st.warning(
                    "Esta variable supera al menos uno de los umbrales "
                    "de drift configurados."
                )
            else:
                st.success(
                    "Esta variable no supera los umbrales configurados."
                )

    # ========================================================
    # 7. TENDENCIA TEMPORAL DEL PSI POR TRIMESTRE
    # ========================================================

    st.subheader("📅 Tendencia temporal del Data Drift")
    st.caption(
        "El gráfico muestra cómo cambia el PSI por trimestre. "
        "Cada trimestre se compara con la misma referencia histórica (80 % inicial)."
    )

    if dates_all.isna().all():
        st.info(
            f"No se puede construir la tendencia porque no hay fechas válidas "
            f"en la columna '{DATE_COLUMN}'."
        )
    else:
        temporal_data = X_all.copy()
        temporal_data["_fecha_analisis"] = pd.to_datetime(
            dates_all, errors="coerce"
        ).to_numpy()
        temporal_data = temporal_data.dropna(subset=["_fecha_analisis"])

        if temporal_data.empty:
            st.info("No hay fechas válidas para construir la tendencia temporal.")
        else:
            temporal_data["_trimestre"] = (
                temporal_data["_fecha_analisis"]
                .dt.to_period("Q")
                .astype(str)
            )

            temporal_features = [
                feature for feature in X_ref.columns
                if feature in X_all.columns
            ]

            temporal_feature = st.selectbox(
                "Selecciona una variable para analizar su PSI en el tiempo",
                temporal_features,
                key="variable_psi_trimestral"
            )

            temporal_points = []
            for quarter, quarter_data in temporal_data.groupby(
                "_trimestre", sort=True
            ):
                psi_value = calculate_psi(
                    X_ref[temporal_feature],
                    quarter_data[temporal_feature]
                )
                temporal_points.append({
                    "Trimestre": quarter,
                    "PSI": psi_value,
                    "Registros": len(quarter_data)
                })

            trend_df = pd.DataFrame(temporal_points).dropna(subset=["PSI"])

            if trend_df.empty:
                st.info("No hay suficientes datos para calcular el PSI por trimestre.")
            else:
                fig_trend, ax_trend = plt.subplots(figsize=(10, 4))
                ax_trend.plot(
                    trend_df["Trimestre"],
                    trend_df["PSI"],
                    marker="o",
                    linewidth=2,
                    label="PSI trimestral"
                )
                ax_trend.axhline(
                    y=0.10,
                    linestyle="--",
                    linewidth=1,
                    label="PSI = 0.10 (cambio moderado)"
                )
                ax_trend.axhline(
                    y=PSI_THRESHOLD,
                    linestyle="--",
                    linewidth=1,
                    label=f"PSI = {PSI_THRESHOLD:.2f} (cambio importante)"
                )
                ax_trend.set_xlabel("Trimestre")
                ax_trend.set_ylabel("PSI")
                ax_trend.set_title(
                    f"Evolución trimestral del PSI: {temporal_feature}"
                )
                ax_trend.tick_params(axis="x", rotation=45)
                ax_trend.grid(axis="y", alpha=0.25)
                ax_trend.legend()
                fig_trend.tight_layout()
                st.pyplot(fig_trend)
                plt.close(fig_trend)

    # ========================================================
    # 8. RECOMENDACIONES
    # ========================================================

    st.subheader("Recomendaciones")

    if drift_count > DRIFT_ALERT_THRESHOLD:
        st.markdown(
            """
            - Revisar las variables señaladas y sus distribuciones.
            - Investigar cambios recientes en la captura o procedencia de datos.
            - Verificar si el cambio tiene una explicación de negocio.
            - Evaluar el rendimiento del modelo con etiquetas reales recientes
              cuando estén disponibles.
            - Considerar una nueva validación o reentrenamiento solo después
              de confirmar el impacto del cambio.
            """
        )
    elif drift_count > 0:
        st.markdown(
            """
            - Inspeccionar las variables con drift detectado.
            - Confirmar si el cambio es esperado o corresponde a un problema
              en la calidad de los datos.
            - Continuar el monitoreo en las siguientes actualizaciones.
            """
        )
    else:
        st.markdown(
            """
            - Continuar con el monitoreo periódico.
            - Revisar la calidad de los datos y la estabilidad de las variables.
            - Recordar que la ausencia de drift detectado no garantiza por sí
              sola que el modelo conserve su rendimiento predictivo.
            """
        )

except Exception as error:
    st.error(
        "No fue posible ejecutar el monitoreo. Revisa el mensaje de error "
        "y la carga de datos."
    )
    st.exception(error)
