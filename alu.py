import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import List, Dict
from langchain_community.vectorstores import AzureSearch
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Sistema RAG con Detector de Alucinaciones",
    layout="wide"
)

# Cargar variables de entorno
load_dotenv()

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

def get_retrieval_score(doc) -> float:
    """Extrae el score de recuperación de un documento de manera segura."""
    try:
        # Intenta obtener el score del documento
        score = getattr(doc, 'score', None)
        if score is not None:
            return float(score)
        return 0.5  # valor por defecto si no hay score
    except (ValueError, TypeError):
        return 0.5

def calculate_mean_score(scores: List[float]) -> float:
    """Calcula la media de los scores de manera segura."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def initialize_system():
    """Inicializa el sistema RAG y lo almacena en la sesión."""
    try:
        # Configuración de Azure
        azure_config = {
            'search_endpoint': os.getenv('AZURE_SEARCH_ENDPOINT'),
            'search_key': os.getenv('AZURE_SEARCH_KEY'),
            'openai_endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),
            'openai_api_key': os.getenv('AZURE_OPENAI_API_KEY'),
            'openai_deployment': os.getenv('AZURE_OPENAI_DEPLOYMENT'),
            'openai_api_version': os.getenv('OPENAI_API_VERSION'),
            'search_index': os.getenv('AZURE_SEARCH_INDEX')
        }

        if not all(azure_config.values()):
            st.error("Error: Algunas variables de entorno no están configuradas.")
            st.stop()

        # Configurar Vector Store
        vector_store = AzureSearch(
            azure_search_endpoint=azure_config['search_endpoint'],
            azure_search_key=azure_config['search_key'],
            index_name=azure_config['search_index'],
            embedding_function=OllamaEmbeddings(model='nomic-embed-text:latest')
        )
        st.session_state.retriever = vector_store.as_retriever(k=1)

        # Configurar LLM
        llm = AzureChatOpenAI(
            azure_deployment=azure_config['openai_deployment'],
            azure_endpoint=azure_config['openai_endpoint'],
            api_key=azure_config['openai_api_key'],
            api_version=azure_config['openai_api_version'],
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )

        # Configurar RAG Chain
        prompt = hub.pull("rlm/rag-prompt")
        st.session_state.rag_chain = prompt | llm | StrOutputParser()

        # Configurar Hallucination Grader
        structured_llm_grader = llm.with_structured_output(GradeHallucinations)
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
                   Give a binary score True or False. True means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ])
        st.session_state.hallucination_grader = hallucination_prompt | structured_llm_grader

        # Crear DataFrame con tipos específicos
        st.session_state.results_db = pd.DataFrame({
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'question': pd.Series(dtype='str'),
            'generation': pd.Series(dtype='str'),
            'is_hallucination': pd.Series(dtype='bool'),
            'retrieval_score': pd.Series(dtype='float64'),
            'category': pd.Series(dtype='str')
        })

        try:
            # Intentar cargar datos existentes con tipos específicos
            loaded_df = pd.read_csv('rag_results.csv', 
                                  dtype={
                                      'question': 'str',
                                      'generation': 'str',
                                      'is_hallucination': 'bool',
                                      'retrieval_score': 'float64',
                                      'category': 'str'
                                  },
                                  parse_dates=['timestamp'])
            st.session_state.results_db = loaded_df
        except FileNotFoundError:
            pass  # Usar el DataFrame vacío creado anteriormente

        st.session_state.initialized = True
        return True

    except Exception as e:
        st.error(f"Error al inicializar el sistema: {str(e)}")
        return False

def evaluate_query(question: str, category: str = None):
    """Evalúa una consulta y registra los resultados."""
    try:
        # Recuperar documentos
        docs = st.session_state.retriever.invoke(question)
        
        # Generar respuesta
        generation = st.session_state.rag_chain.invoke({
            "context": "\n\n".join(doc.page_content for doc in docs),
            "question": question
        })
        
        # Evaluar alucinación
        grader_result = st.session_state.hallucination_grader.invoke({
            "documents": docs,
            "generation": generation
        })
        
        # Calcular score de recuperación de manera segura
        scores = [get_retrieval_score(doc) for doc in docs]
        retrieval_score = calculate_mean_score(scores)
        
        # Crear nuevo registro con tipos específicos
        new_row = pd.DataFrame([{
            'timestamp': pd.Timestamp.now(),
            'question': str(question),
            'generation': str(generation),
            'is_hallucination': bool(not grader_result.binary_score),
            'retrieval_score': float(retrieval_score),
            'category': str(category) if category else 'General'
        }])
        
        # Concatenar y guardar resultados
        st.session_state.results_db = pd.concat([st.session_state.results_db, new_row], ignore_index=True)
        st.session_state.results_db.to_csv('rag_results.csv', index=False)
        
        return {
            'generation': generation,
            'is_hallucination': not grader_result.binary_score,
            'retrieval_score': retrieval_score,
            'docs': docs
        }
    except Exception as e:
        st.error(f"Error al evaluar la consulta: {str(e)}")
        return None

def main():
    st.title("Sistema RAG con Detector de Alucinaciones")
    
    # Inicializar el sistema si no está inicializado
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        with st.spinner("Inicializando el sistema..."):
            if not initialize_system():
                st.stop()
    
    # Interface de usuario
    with st.form("query_form"):
        question = st.text_input("Ingresa tu pregunta:")
        category = st.selectbox(
            "Categoría:",
            ["General", "Técnico", "Histórico", "Otro"]
        )
        submitted = st.form_submit_button("Evaluar")
    
    if submitted and question:
        with st.spinner("Procesando consulta..."):
            result = evaluate_query(question, category)
            if result:
                st.success("Consulta procesada exitosamente")
                
                # Mostrar resultados
                st.subheader("Resultados")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Respuesta:**")
                    st.write(result['generation'])
                with col2:
                    st.metric(
                        label="Estado de Alucinación",
                        value="Alucinación" if result['is_hallucination'] else "Respuesta Fundamentada"
                    )
                    st.metric(
                        label="Score de Recuperación",
                        value=f"{result['retrieval_score']:.2f}"
                    )
    
    # Mostrar métricas si hay datos
    if hasattr(st.session_state, 'results_db') and not st.session_state.results_db.empty:
        st.subheader("Métricas del Sistema")
        
        # Calcular métricas de manera segura 
        total_queries = len(st.session_state.results_db)
        hallucination_rate = st.session_state.results_db['is_hallucination'].mean() * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Consultas", total_queries)
            st.metric("Tasa de Alucinaciones", f"{hallucination_rate:.2f}%")
        
        with col2:
            try:
                # Gráfico de tendencia
                fig = px.line(
                    st.session_state.results_db,
                    x='timestamp',
                    y='retrieval_score',
                    title="Tendencia de Scores de Recuperación"
                )
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error al generar el gráfico: {str(e)}")

if __name__ == "__main__":
    main()