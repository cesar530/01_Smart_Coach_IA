# 🏋️ Smart Coach AI - Recomendador Inteligente de Rutinas

Sistema de recomendación de rutinas de ejercicio personalizado que combina **Machine Learning clásico (K-Means Clustering)** con **Large Language Models (LLM)** para generar planes de entrenamiento adaptados a cada usuario.

## 🎯 Objetivo del Proyecto

Demostrar la integración de técnicas de ML clásico con modelos de lenguaje avanzados para crear un sistema de recomendación inteligente que:
- Agrupa usuarios con perfiles similares usando clustering
- Genera rutinas personalizadas usando LLMs
- Proporciona explicaciones claras y contextualizadas
- Ofrece una interfaz web interactiva

## 📋 Características

- **Clustering Inteligente**: Agrupa usuarios según experiencia, objetivos, equipo y tiempo disponible
- **Recomendaciones Personalizadas**: Rutinas adaptadas al perfil del usuario
- **Explicaciones Generadas por IA**: El LLM explica el razonamiento detrás de cada recomendación
- **Interfaz Web Interactiva**: Aplicación Streamlit fácil de usar
- **Análisis Exploratorio**: Notebook con visualizaciones y análisis de datos

## 🛠️ Tecnologías

- **Python 3.10+**
- **Scikit-Learn**: Clustering K-Means
- **LangChain**: Framework para integrar LLMs
- **Ollama**: LLM local (llama2, mistral, etc.)
- **Streamlit**: Interfaz web
- **Pandas & NumPy**: Manipulación de datos
- **Matplotlib & Seaborn**: Visualizaciones

## 📁 Estructura del Proyecto

```
01_Smart_Coach_IA/
├── README.md                          # Documentación del proyecto
├── requirements.txt                   # Dependencias
├── .gitignore                        # Archivos ignorados
├── 01_Smart_Coach_Analysis.ipynb    # Notebook de análisis y clustering
├── app.py                            # Aplicación Streamlit
├── coach_recommender.py              # Sistema de recomendación
└── utils.py                          # Funciones auxiliares
```

## 🚀 Instalación

1. **Clonar el repositorio**:
```bash
git clone <tu-repositorio>
cd 01_Smart_Coach_IA
```

2. **Crear entorno virtual**:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Instalar Ollama** (para LLM local):
   - Descargar desde: https://ollama.ai/
   - Instalar el modelo: `ollama pull llama2`

## 📊 Uso

### 1. Análisis Exploratorio (Notebook)

```bash
jupyter notebook 01_Smart_Coach_Analysis.ipynb
```

El notebook incluye:
- Generación de datos sintéticos de usuarios
- Análisis exploratorio de datos (EDA)
- Implementación de K-Means clustering
- Visualización de clusters
- Perfiles de usuarios por cluster

### 2. Aplicación Web

```bash
streamlit run app.py
```

La aplicación permite:
- Ingresar información del usuario (experiencia, objetivos, equipo, tiempo)
- Recibir recomendaciones personalizadas
- Ver explicaciones generadas por el LLM
- Visualizar el cluster al que pertenece

## 💡 Cómo Funciona

1. **Entrada del Usuario**: El usuario proporciona información sobre su perfil
2. **Clustering**: El sistema identifica el cluster más similar usando K-Means
3. **Generación de Contexto**: Se extraen características del cluster
4. **LLM Generation**: El LLM genera una rutina personalizada con explicaciones
5. **Presentación**: Se muestra la rutina completa al usuario

## 📈 Impacto y Aprendizajes

Este proyecto demuestra:
- ✅ **ML Clásico**: Implementación de K-Means clustering
- ✅ **LLMs**: Integración con modelos de lenguaje
- ✅ **Sistemas de Recomendación**: Diseño de recomendadores híbridos
- ✅ **Desarrollo Web**: Creación de aplicaciones interactivas
- ✅ **Pipeline Completo**: Desde datos hasta producción

## 🔮 Mejoras Futuras

- [ ] Integrar base de datos para almacenar perfiles
- [ ] Implementar feedback loop para mejorar recomendaciones
- [ ] Agregar más variables (lesiones, preferencias, etc.)
- [ ] Sistema de progresión y seguimiento
- [ ] Integración con APIs de fitness (Strava, Fitbit, etc.)

## 📝 Licencia

Este proyecto es parte de un portafolio personal y está disponible para propósitos educativos.

## 👤 Autor

**Cesar Adrian Delgado Diaz**
- 💼 LinkedIn: [linkedin.com/in/cesar-delgado-diaz](linkedin.com/in/cesar-delgado-diaz)
- 🐙 GitHub: [github.com/tu-usuario](https://github.com/cesar530)

---

**Tags**: `machine-learning` `llm` `clustering` `recommender-system` `streamlit` `langchain` `python` `fitness` `ai`
