import streamlit as st
import pandas as pd
import math
import numpy as np
from modelo_controller import ModelController



# ==========================

# CONFIGURACIÓN DE LA APP

# ==========================

st.set_page_config(
page_title="Prediccion_ods",
page_icon="📖",
layout="wide"
)



# ==========================

# PANEL DE ENTRADA

# ==========================


st.markdown("""
<div style="
    display: flex;
    align-items: center;
    background-color: #EAEDCE;
    padding: 15px 20px;
    border-radius: 10px;
    color: black;
">
    <img src="https://www.funcionpublica.gov.co/documents/d/guest/logo_uniandes" 
         style='height:50px; margin-right:15px;'>
    <div style="flex: 1; text-align: center; font-size:24px; font-weight:bold;">
        🧾 Microproyecto 2 Prediccion ODS 
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("**ℹ️ Introduce un texto para que el modelo de Machine Learning realice la clasificación automática del Objetivo de Desarrollo Sostenible (ODS) y muestre su probabilidad asociada.**")

#st.sidebar.image("logo_cenit.png", caption="Eficiencia energética", width=160)
st.sidebar.header("🧾 Texto a clasificar")



with st.sidebar.form("prediction_form"):
    
    texto = st.text_area(
        "Escribe tu texto",
        height=180
    )
    
    submit_text = st.form_submit_button("Clasificar")

if submit_text:

    texto_input = texto 
    st.info(f"📥 **Texto recibido:**\n\n{texto_input}")

    modelo = ModelController()
    text, y_pred, y_prob = modelo.predict(texto_input)

    # Extraer valores correctos
    clase_predicha = y_pred[0]
    prob_max = y_prob[0].max()

    # Obtener clases del modelo (si es pipeline)
    clases = modelo.model.classes_

    probs_df = (
        pd.DataFrame({
           "Probabilidad (%)": (y_prob[0] * 100)
        },  index= clases,)
        .sort_values(by="Probabilidad (%)", ascending=False)
        .head(5)
        .round(2)
    )

    st.dataframe(
        probs_df.style.hide(axis="index"),
        use_container_width=True
    )


    col1, col2 = st.columns([1, 1])  

    with col1:
        st.caption("🗣 El ODS Predicho es")
        st.metric("Predicción", clase_predicha)

    with col2:
        st.caption("📊 Confianza del modelo")
        st.metric("Probabilidad", f"{prob_max*100:.2f}%")



