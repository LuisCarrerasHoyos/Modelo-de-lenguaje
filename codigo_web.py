import streamlit as st

# ────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA (DEBE SER PRIMERO)
# ────────────────────────────────────────

st.set_page_config(
    page_title="Asistente Farmacéutico AEMPS",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS optimizado
st.markdown("""
    <style>
        .stTextInput input {font-size: 16px; padding: 10px;}
        .stButton button {background-color: #4CAF50; color: white; padding: 8px 20px;}
        .response-box {padding: 15px; background: #f8f9fa; border-radius: 8px; margin-top: 15px;}
        .title {color: #2c3e50; text-align: center; margin-bottom: 20px;}
        .footer {text-align: center; margin-top: 30px; color: #6c757d; font-size: 12px;}
    </style>
""", unsafe_allow_html=True)

# Título simplificado
st.markdown('<h1 class="title">💊 Asistente Farmacéutico</h1>', unsafe_allow_html=True)
st.markdown("Consulta información técnica sobre los siguientes medicamentos autorizados por la AEMPS: EXJADE, LUMINITY, SPRYCEL, TANDEMACT, DIACOMIT, PREZISTA ")

# ────────────────────────────────────────
# IMPORTS Y CONFIGURACIÓN DEL MODELO
# ────────────────────────────────────────

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import pandas as pd
import requests
import zipfile
import lxml.etree as ET
import json
import re
import os
import pickle
import numpy as np
from langgraph.graph import StateGraph, END
from typing import TypedDict
import language_tool_python
import time
from sklearn.metrics.pairwise import cosine_similarity

# Cargar variables de entorno
@st.cache_resource
def load_environment():
    load_dotenv()
    return True

load_environment()

# Configuración de modelos (cached)
@st.cache_resource
def setup_models():
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0, max_tokens=50000)
    return embedding_model, llm

embedding_model, llm = setup_models()




# ────────────────────────────────────────
# DESCARGAR Y PROCESAR DATOS (cached)
# ────────────────────────────────────────

@st.cache_data(ttl=24*60*60)  # Cache por 24 horas
def download_and_process_data():
    # URL del archivo de medicamentos
    URL_XLS = "https://listadomedicamentos.aemps.gob.es/medicamentos.xls"
    
    # Descargar archivo
    with requests.Session() as session:
        response = session.get(URL_XLS)
        with open("medicamentos.xls", "wb") as f:
            f.write(response.content)
    
    # Procesar medicamentos
    medicamentos = pd.read_excel('medicamentos.xls')
    
    # Crear nombre abreviado
    medicamentos['Nombre'] = medicamentos['Medicamento'].str.split(' ', expand=True)[0]
    
    # Para los que empiezan por "ácido" usar las dos primeras palabras
    acidos = medicamentos[medicamentos['Nombre'].str.lower() == 'acido']
    if not acidos.empty:
        acidos['Nombre'] = acidos['Medicamento'].str.split().apply(lambda x: ' '.join(x[:2]))
        medicamentos.update(acidos)
    
    # Filtrar solo los medicamentos que necesitas
    medicamentos_filtrados = medicamentos[
        medicamentos['Nombre'].str.upper().isin(['EXJADE', 'LUMINITY', 'SPRYCEL', 'TANDEMACT', 'DIACOMIT', 'PREZISTA'])
    ].copy()
    
    return medicamentos_filtrados

# Llamar a la función para obtener los datos
medicamentos_filtrados = download_and_process_data()

# ────────────────────────────────────────
# VECTORSTORE (cached)
# ────────────────────────────────────────

@st.cache_resource
def setup_vectorstore():
    faiss_path = "faiss_index"
    metadata_path = "documentos_metadata.pkl"
    
    if os.path.exists(faiss_path):
        vectordb = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
        with open(metadata_path, "rb") as f:
            documentos = pickle.load(f)
    else:
        documentos = []
        for _, row in medicamentos_filtrados.iterrows():
            doc = Document(page_content="Sample content", metadata=row.to_dict())
            documentos.append(doc)
        
        vectordb = FAISS.from_documents(documentos, embedding_model)
        vectordb.save_local(faiss_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(documentos, f)
    
    return vectordb, documentos

vectordb, documentos = setup_vectorstore()

# ────────────────────────────────────────
# FUNCIONES DEL GRAFO (optimizadas)
# ────────────────────────────────────────

class GraphState(TypedDict):
    query: str
    pregunta_valida: bool
    respuesta: str
    respuesta_valida: bool
    medicamentos_text: str

# Ahora que medicamentos_filtrados está definido, podemos usarlo
nombres_validos = set(nombre.split()[0].lower() for nombre in medicamentos_filtrados['Medicamento'])

def calcular_embeddings(query, embedding_model):
    return np.array(embedding_model.embed_query(query)).reshape(1, -1)

def obtener_vectores_faiss(vectordb):
    return vectordb.index.reconstruct_n(0, vectordb.index.ntotal)

def buscar_documentos_similares_cosine(query, documentos, vectores, embedding_model, k=3):
    query_embedding = calcular_embeddings(query, embedding_model)
    similitudes = cosine_similarity(query_embedding, vectores)[0]
    top_indices = np.argsort(similitudes)[-k:][::-1]
    return [documentos[i] for i in top_indices]

def validar_pregunta(state: GraphState) -> GraphState:
    pregunta = state["query"]
    tool = language_tool_python.LanguageTool('es', remote_server='https://api.languagetool.org')
    matches = tool.check(pregunta)
    
    ortografia_errores = [match for match in matches 
                         if match.ruleIssueType == 'misspelling' 
                         and pregunta[match.offset:match.offset+match.errorLength].lower() not in nombres_validos]
    
    if ortografia_errores:
        pregunta_corregida = pregunta
        for error in reversed(ortografia_errores):
            if error.replacements:
                start, end = error.offset, error.offset + error.errorLength
                pregunta_corregida = pregunta_corregida[:start] + error.replacements[0] + pregunta_corregida[end:]
        
        return {
            **state,
            "pregunta_valida": False,
            "query": pregunta_corregida,
            "motivo": 'La pregunta contiene errores ortográficos y ha sido corregida.'
        }
    return {
        **state,
        "pregunta_valida": True,
        "motivo": 'La pregunta está correctamente escrita.'
    }

def responder_consulta_return(query, vectordb, documentos, k=3):
    vectores = obtener_vectores_faiss(vectordb)
    docs_relevantes = buscar_documentos_similares_cosine(query, documentos, vectores, embedding_model, k)

    medicamentos_text = "Información relevante de medicamentos:\n\n"
    for doc in docs_relevantes:
        row = doc.metadata
        texto = f"Medicamento: {row['Medicamento']}, Principios Activos: {row['Principios Activos']}\n"
        texto += f"Sección relevante:\n{doc.page_content.strip()}\n"
        medicamentos_text += texto + "\n"

    prompt = f"""
    Responde usando ÚNICAMENTE esta información:
    {medicamentos_text}
    Instrucciones importantes:
    1. Sé preciso y conciso.
    2. Si no hay información relevante, indica que no dispones de esos datos.
    3. Nunca inventes información.
    4. Cita solo los medicamentos con información relevante.
    5. Resume la información, no la copies.
    6. No me digas en que sección se encuentra la información, solo dame la respuesta.
    {query}
    """
    response = llm.invoke(prompt)
    return response.content, medicamentos_text

def responder(state: GraphState) -> GraphState:
    query = state["query"]
    respuesta, medicamentos_text = responder_consulta_return(query, vectordb, documentos)
    return {**state, "respuesta": respuesta, "medicamentos_text": medicamentos_text}

def validar_respuesta(state: GraphState) -> GraphState:
    query = state["query"]
    respuesta = state["respuesta"]
    medicamentos_text = state["medicamentos_text"]

    prompt = f"""
    Evalúa si esta respuesta cumple:
    1. Coherente con la información
    2. Precisa y directa
    3. Concisa
    4. No copia literal
    
    Info: {medicamentos_text}
    Pregunta: {query}
    Respuesta: {respuesta}
    
    Responde solo con "True" o "False".
    """
    evaluation = llm.invoke(prompt).content.strip()

    if evaluation == "True":
        return {**state, "respuesta_valida": True}
    
    correction_prompt = f"""
    Mejora esta respuesta para que sea:
    1. Coherente
    2. Precisa
    3. Resumen claro
    
    Info: {medicamentos_text}
    Pregunta: {query}
    Respuesta original: {respuesta}
    
    Devuelve solo la respuesta corregida.
    """
    corrected_response = llm.invoke(correction_prompt).content.strip()
    return {**state, "respuesta": corrected_response, "respuesta_valida": False}

# Configurar el grafo
graph = StateGraph(GraphState)
graph.add_node("validar_pregunta", validar_pregunta)
graph.add_node("responder", responder)
graph.add_node("validar_respuesta", validar_respuesta)
graph.set_entry_point("validar_pregunta")
graph.add_edge("validar_pregunta", "responder")
graph.add_edge("responder", "validar_respuesta")
graph.add_edge("validar_respuesta", END)
app = graph.compile()

# ────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ────────────────────────────────────────

# Estado de la sesión para mantener el historial
if 'history' not in st.session_state:
    st.session_state.history = []

# Interfaz principal
query = st.text_input(
    "Escribe tu pregunta sobre un medicamento:",
    placeholder="Ej: ¿Afecta el medicamento EXJADE a la conducción?",
    key="query_input"
)

if st.button("Consultar", key="submit_button") and query:
    with st.spinner('Procesando tu consulta...'):
        start_time = time.time()
        
        try:
            entrada = {"query": query}
            salida = app.invoke(entrada)
            
            # Mostrar respuesta
            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            st.markdown("**Respuesta:**")
            st.markdown(salida['respuesta'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Guardar en historial
            st.session_state.history.append((query, salida['respuesta']))
            
        except Exception as e:
            st.error(f"Error al procesar tu consulta: {str(e)}")
        
        st.info(f"Tiempo de procesamiento: {time.time()-start_time:.2f} segundos")

# Historial de consultas
if st.session_state.history:
    with st.expander("Historial de consultas"):
        for q, a in reversed(st.session_state.history):
            st.markdown(f"**Pregunta:** {q}")
            st.markdown(f"**Respuesta:** {a}")
            st.markdown("---")

# Pie de página
st.markdown("""
    <div class="footer">
        <p>Datos proporcionados por la AEMPS</p>
        <p>No sustituye el consejo médico profesional</p>
    </div>
""", unsafe_allow_html=True)