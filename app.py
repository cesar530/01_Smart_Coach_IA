"""
Smart Coach AI - Aplicación Web Streamlit
Interfaz interactiva para el sistema de recomendación de rutinas
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from coach_recommender import SmartCoachRecommender
import pickle

# Configuración de la página
st.set_page_config(
    page_title="Smart Coach AI",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def cargar_recommender():
    """Carga el sistema de recomendación (cached)."""
    try:
        return SmartCoachRecommender()
    except Exception as e:
        st.error(f"❌ Error al cargar el sistema: {str(e)}")
        st.info("💡 Asegúrate de haber ejecutado el notebook de análisis primero para generar los modelos.")
        return None


def mostrar_header():
    """Muestra el header de la aplicación."""
    st.markdown('<p class="main-header">🏋️ Smart Coach AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Tu Entrenador Personal Inteligente</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ---
    Bienvenido a **Smart Coach AI**, un sistema de recomendación que combina:
    - 🤖 **Machine Learning** (K-Means Clustering) para agrupar perfiles similares
    - 🧠 **Large Language Models** para generar rutinas personalizadas
    - 📊 **Análisis de datos** para identificar patrones y tendencias
    """)


def formulario_usuario():
    """Crea el formulario para capturar información del usuario."""
    st.sidebar.header("📋 Tu Perfil")
    
    with st.sidebar.form("perfil_usuario"):
        st.subheader("Información Básica")
        
        experiencia = st.selectbox(
            "Nivel de Experiencia",
            options=['Principiante', 'Intermedio', 'Avanzado'],
            help="Selecciona tu nivel de experiencia en entrenamiento"
        )
        
        edad = st.slider(
            "Edad",
            min_value=18,
            max_value=100,
            value=30,
            help="Tu edad actual"
        )
        
        st.markdown("---")
        st.subheader("Objetivos y Disponibilidad")
        
        objetivo = st.selectbox(
            "Objetivo Principal",
            options=['Pérdida de peso', 'Ganar músculo', 'Resistencia', 'Salud general'],
            help="¿Qué quieres lograr con tu entrenamiento?"
        )
        
        equipo = st.selectbox(
            "Equipo Disponible",
            options=['Gimnasio completo', 'Casa con equipo', 'Solo peso corporal'],
            help="¿Qué equipo tienes disponible?"
        )
        
        tiempo = st.slider(
            "Tiempo por Sesión (minutos)",
            min_value=15,
            max_value=120,
            value=45,
            step=5,
            help="¿Cuánto tiempo puedes dedicar por sesión?"
        )
        
        frecuencia = st.slider(
            "Frecuencia Semanal (días)",
            min_value=1,
            max_value=7,
            value=3,
            help="¿Cuántos días a la semana puedes entrenar?"
        )
        
        st.markdown("---")
        st.subheader("Restricciones Físicas")
        
        restricciones = st.selectbox(
            "¿Tienes alguna lesión o restricción?",
            options=['Ninguna', 'Rodilla', 'Espalda', 'Hombro', 'Muñeca'],
            help="Esto nos ayudará a adaptar los ejercicios"
        )
        
        submitted = st.form_submit_button("🚀 Generar Mi Rutina", use_container_width=True)
        
        if submitted:
            usuario_info = {
                'experiencia': experiencia,
                'objetivo': objetivo,
                'equipo': equipo,
                'tiempo_disponible': tiempo,
                'frecuencia_semanal': frecuencia,
                'edad': edad,
                'restricciones': restricciones
            }
            return usuario_info
    
    return None


def mostrar_cluster_info(coach, cluster_id):
    """Muestra información del cluster asignado."""
    cluster_info = coach.obtener_info_cluster(cluster_id)
    
    if cluster_info is None:
        return
    
    st.markdown(f"""
    <div class="info-box">
        <h3>👥 Tu Grupo de Entrenamiento: Cluster {cluster_id}</h3>
        <p>Has sido asignado a un grupo de <strong>{cluster_info['n_usuarios']} usuarios</strong> con perfil similar al tuyo.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "⏱️ Tiempo Promedio",
            f"{cluster_info['tiempo_promedio']:.0f} min",
            delta=None
        )
    
    with col2:
        st.metric(
            "📅 Frecuencia Promedio",
            f"{cluster_info['frecuencia_promedio']:.1f} días",
            delta=None
        )
    
    with col3:
        st.metric(
            "👤 Edad Promedio",
            f"{cluster_info['edad_promedio']:.0f} años",
            delta=None
        )
    
    with col4:
        st.metric(
            "👥 Usuarios Similares",
            cluster_info['n_usuarios'],
            delta=None
        )
    
    # Características comunes del cluster
    st.markdown("### 🎯 Características Comunes del Grupo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        **Experiencia Común:**  
        {cluster_info['experiencia_comun']}
        """)
    
    with col2:
        st.markdown(f"""
        **Objetivo Común:**  
        {cluster_info['objetivo_comun']}
        """)
    
    with col3:
        st.markdown(f"""
        **Equipo Común:**  
        {cluster_info['equipo_comun']}
        """)


def mostrar_rutina(resultado):
    """Muestra la rutina generada."""
    st.markdown("---")
    st.markdown("## 📝 Tu Rutina Personalizada")
    
    # Mostrar la rutina
    st.markdown(resultado['rutina'])
    
    # Recomendaciones adicionales
    if resultado['recomendaciones_adicionales']:
        st.markdown("---")
        st.markdown("## 💡 Recomendaciones Adicionales")
        
        for rec in resultado['recomendaciones_adicionales']:
            st.markdown(f"- {rec}")
    
    # Botón de descarga
    st.markdown("---")
    st.download_button(
        label="📥 Descargar Rutina (Markdown)",
        data=resultado['rutina'],
        file_name="mi_rutina_smart_coach.md",
        mime="text/markdown"
    )


def mostrar_visualizacion_clusters(coach):
    """Muestra visualización de todos los clusters."""
    st.markdown("---")
    st.markdown("## 📊 Análisis de Clusters")
    
    # Cargar el dataset
    df = coach.df
    
    # Tabs para diferentes visualizaciones
    tab1, tab2, tab3 = st.tabs(["📈 Distribución", "🎯 Características", "📋 Tabla Resumen"])
    
    with tab1:
        # Gráfico de distribución de usuarios por cluster
        cluster_counts = df['cluster'].value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f"Cluster {i}" for i in cluster_counts.index],
                y=cluster_counts.values,
                marker_color=['#1E88E5', '#43A047', '#FB8C00', '#E53935']
            )
        ])
        
        fig.update_layout(
            title="Distribución de Usuarios por Cluster",
            xaxis_title="Cluster",
            yaxis_title="Número de Usuarios",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Características promedio por cluster
        cluster_stats = df.groupby('cluster').agg({
            'tiempo_disponible': 'mean',
            'frecuencia_semanal': 'mean',
            'edad': 'mean'
        }).round(1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Tiempo (min)',
            x=[f"Cluster {i}" for i in cluster_stats.index],
            y=cluster_stats['tiempo_disponible'],
            marker_color='#1E88E5'
        ))
        
        fig.add_trace(go.Bar(
            name='Frecuencia (días)',
            x=[f"Cluster {i}" for i in cluster_stats.index],
            y=cluster_stats['frecuencia_semanal'] * 10,  # Escalar para visualización
            marker_color='#43A047'
        ))
        
        fig.add_trace(go.Bar(
            name='Edad (años)',
            x=[f"Cluster {i}" for i in cluster_stats.index],
            y=cluster_stats['edad'],
            marker_color='#FB8C00'
        ))
        
        fig.update_layout(
            title="Características Promedio por Cluster",
            xaxis_title="Cluster",
            yaxis_title="Valor",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("📝 Nota: La frecuencia está escalada ×10 para mejor visualización")
    
    with tab3:
        # Tabla resumen
        st.dataframe(
            coach.listar_clusters(),
            use_container_width=True,
            hide_index=True
        )


def main():
    """Función principal de la aplicación."""
    
    # Header
    mostrar_header()
    
    # Cargar sistema de recomendación
    with st.spinner("🔄 Cargando sistema de recomendación..."):
        coach = cargar_recommender()
    
    if coach is None:
        st.stop()
    
    # Sidebar con formulario
    usuario_info = formulario_usuario()
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["🏋️ Mi Rutina", "📊 Explorar Clusters", "ℹ️ Acerca de"])
    
    with tab1:
        if usuario_info is None:
            st.info("👈 Completa tu perfil en la barra lateral para generar tu rutina personalizada.")
            
            # Mostrar ejemplo
            st.markdown("---")
            st.markdown("### 🎯 ¿Cómo funciona?")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                #### 1️⃣ Completa tu Perfil
                Proporciona información sobre:
                - Tu experiencia
                - Tus objetivos
                - Equipo disponible
                - Tiempo y frecuencia
                """)
            
            with col2:
                st.markdown("""
                #### 2️⃣ IA Analiza
                El sistema:
                - Te agrupa con usuarios similares
                - Analiza patrones exitosos
                - Identifica tu perfil ideal
                """)
            
            with col3:
                st.markdown("""
                #### 3️⃣ Recibe tu Rutina
                Obtienes:
                - Rutina personalizada
                - Explicaciones detalladas
                - Recomendaciones extras
                """)
        
        else:
            # Generar recomendación
            with st.spinner("🤖 Generando tu rutina personalizada... Esto puede tomar unos segundos."):
                resultado = coach.recomendar(usuario_info)
            
            if resultado['error']:
                st.error(f"❌ Error: {resultado['mensaje']}")
                for error in resultado['errores']:
                    st.markdown(f"- {error}")
            else:
                # Mostrar información del cluster
                mostrar_cluster_info(coach, resultado['cluster_id'])
                
                # Mostrar la rutina
                mostrar_rutina(resultado)
                
                # Mensaje de éxito
                st.markdown("""
                <div class="success-box">
                    <h4>✅ ¡Rutina generada exitosamente!</h4>
                    <p>Recuerda: la constancia es clave. Sigue tu rutina y ajusta según tus progresos.</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📊 Explora los Clusters Disponibles")
        st.markdown("Descubre los diferentes perfiles de usuarios y sus características.")
        mostrar_visualizacion_clusters(coach)
    
    with tab3:
        st.markdown("### ℹ️ Acerca de Smart Coach AI")
        
        st.markdown("""
        **Smart Coach AI** es un proyecto de portafolio que demuestra la integración de:
        
        #### 🔧 Tecnologías Utilizadas
        - **Python**: Lenguaje principal
        - **Scikit-Learn**: K-Means Clustering para agrupar usuarios
        - **LangChain**: Framework para integración de LLMs
        - **Ollama**: LLM local para generación de rutinas
        - **Streamlit**: Interfaz web interactiva
        - **Pandas & NumPy**: Manipulación de datos
        - **Plotly**: Visualizaciones interactivas
        
        #### 🎯 Metodología
        1. **Clustering**: Agrupa usuarios con características similares usando K-Means
        2. **Análisis de Perfil**: Extrae características comunes del cluster
        3. **Generación con LLM**: Crea rutinas personalizadas usando modelos de lenguaje
        4. **Recomendaciones**: Proporciona consejos adicionales basados en el perfil
        
        #### 📈 Impacto y Aprendizajes
        Este proyecto demuestra:
        - ✅ Implementación de algoritmos de ML clásicos
        - ✅ Integración con Large Language Models
        - ✅ Diseño de sistemas de recomendación
        - ✅ Desarrollo de aplicaciones web interactivas
        - ✅ Pipeline completo: desde datos hasta producción
        
        #### 👤 Autor
        **Cesar Delgado**  
        Data Scientist | ML Engineer
        
        ---
        
        💡 **Nota**: Este proyecto utiliza Ollama con modelos locales. Asegúrate de tener Ollama instalado
        y el modelo descargado (`ollama pull llama2`).
        
        🔗 **GitHub**: [Repositorio del proyecto](#)
        """)


if __name__ == "__main__":
    main()
