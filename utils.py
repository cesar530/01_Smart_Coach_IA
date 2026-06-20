"""
Utilidades para el Smart Coach AI
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def crear_prompt_rutina(usuario_info: Dict, cluster_profile: Dict) -> str:
    """
    Crea un prompt detallado para el LLM basado en el perfil del usuario y su cluster.
    
    Args:
        usuario_info: Información del usuario individual
        cluster_profile: Perfil promedio del cluster al que pertenece
        
    Returns:
        Prompt formateado para el LLM
    """
    prompt = f"""
Eres un entrenador personal experto. 
Crea una rutina de ejercicio personalizada basada en el siguiente perfil:

INFORMACIÓN DEL USUARIO:
- Nivel de experiencia: {usuario_info['experiencia']}
- Objetivo principal: {usuario_info['objetivo']}
- Equipo disponible: {usuario_info['equipo']}
- Tiempo por sesión: {usuario_info['tiempo_disponible']} minutos
- Frecuencia semanal: {usuario_info['frecuencia_semanal']} días
- Edad: {usuario_info['edad']} años
- Restricciones físicas: {usuario_info['restricciones']}

PERFIL DEL GRUPO (usuarios similares):
- Tiempo promedio: {cluster_profile['tiempo_promedio']:.0f} minutos
- Frecuencia promedio: {cluster_profile['frecuencia_promedio']:.1f} días/semana
- Edad promedio: {cluster_profile['edad_promedio']:.0f} años
- Objetivo común: {cluster_profile['objetivo_comun']}

INSTRUCCIONES:
1. Diseña una rutina completa adaptada a las características del usuario
2. Considera las restricciones físicas mencionadas
3. Ajusta los ejercicios al equipo disponible
4. Respeta el tiempo disponible por sesión diaria
5. Incluye calentamiento y enfriamiento
6. Proporciona sets, repeticiones y tiempos de descanso específicos
7. Explica POR QUÉ cada ejercicio es adecuado para este usuario
8. La rutina debe ser segura y efectiva
9. La rutina debe tener de 2 a 6 ejercicios por sesión
10. La rutina debe estar en idioma español

FORMATO DE RESPUESTA:
Organiza la rutina por días de la semana, con la siguiente estructura para cada día:

**DÍA X - [Enfoque del día]**

🔥 Calentamiento (X minutos):
- [Ejercicio 1]
- [Ejercicio 2]
- ...

💪 Entrenamiento Principal:
1. [Ejercicio]: X sets × X reps (descanso: X seg)
   → Explicación de por qué este ejercicio
2. [Ejercicio]: X sets × X reps (descanso: X seg)
   → Explicación
3. ...

❄️ Enfriamiento (X minutos):
- [Estiramientos específicos]

📝 NOTAS IMPORTANTES:
- [Consejos personalizados]
- [Precauciones según restricciones]

🎯 PROGRESIÓN:
Explica cómo el usuario puede progresar en 2-4 semanas.
"""
    return prompt


def extraer_perfil_cluster(df: pd.DataFrame, cluster_id: int) -> Dict:
    """
    Extrae el perfil característico de un cluster específico.
    
    Args:
        df: DataFrame con usuarios y sus clusters
        cluster_id: ID del cluster
        
    Returns:
        Diccionario con el perfil del cluster
    """
    cluster_data = df[df['cluster'] == cluster_id]
    
    perfil = {
        'cluster_id': cluster_id,
        'n_usuarios': len(cluster_data),
        'tiempo_promedio': cluster_data['tiempo_disponible'].mean(),
        'frecuencia_promedio': cluster_data['frecuencia_semanal'].mean(),
        'edad_promedio': cluster_data['edad'].mean(),
        'experiencia_comun': cluster_data['experiencia'].mode()[0],
        'objetivo_comun': cluster_data['objetivo'].mode()[0],
        'equipo_comun': cluster_data['equipo'].mode()[0],
        'restriccion_comun': cluster_data['restricciones'].mode()[0]
    }
    
    return perfil


def predecir_cluster(modelo_kmeans, scaler, label_encoders, usuario_info: Dict) -> int:
    """
    Predice el cluster al que pertenece un nuevo usuario.
    
    Args:
        modelo_kmeans: Modelo KMeans entrenado
        scaler: StandardScaler entrenado
        label_encoders: Diccionario con encoders para variables categóricas
        usuario_info: Información del nuevo usuario
        
    Returns:
        ID del cluster predicho
    """
    # Codificar variables categóricas
    experiencia_encoded = label_encoders['experiencia'].transform([usuario_info['experiencia']])[0]
    objetivo_encoded = label_encoders['objetivo'].transform([usuario_info['objetivo']])[0]
    equipo_encoded = label_encoders['equipo'].transform([usuario_info['equipo']])[0]
    
    # Crear vector de features
    features = np.array([[
        experiencia_encoded,
        objetivo_encoded,
        equipo_encoded,
        usuario_info['tiempo_disponible'],
        usuario_info['frecuencia_semanal'],
        usuario_info['edad']
    ]])
    
    # Normalizar y predecir
    features_scaled = scaler.transform(features)
    cluster_id = modelo_kmeans.predict(features_scaled)[0]
    
    return cluster_id


def validar_input_usuario(usuario_info: Dict) -> Tuple[bool, List[str]]:
    """
    Valida que la información del usuario esté completa y sea válida.
    
    Args:
        usuario_info: Información del usuario
        
    Returns:
        Tupla (es_valido, lista_errores)
    """
    errores = []
    
    # Validar campos requeridos
    campos_requeridos = ['experiencia', 'objetivo', 'equipo', 'tiempo_disponible', 
                         'frecuencia_semanal', 'edad', 'restricciones']
    
    for campo in campos_requeridos:
        if campo not in usuario_info:
            errores.append(f"Falta el campo requerido: {campo}")
    
    # Validar rangos numéricos
    if 'tiempo_disponible' in usuario_info:
        if not (15 <= usuario_info['tiempo_disponible'] <= 120):
            errores.append("Tiempo disponible debe estar entre 15 y 120 minutos")
    
    if 'frecuencia_semanal' in usuario_info:
        if not (1 <= usuario_info['frecuencia_semanal'] <= 7):
            errores.append("Frecuencia semanal debe estar entre 1 y 7 días")
    
    if 'edad' in usuario_info:
        if not (18 <= usuario_info['edad'] <= 100):
            errores.append("Edad debe estar entre 18 y 100 años")
    
    # Validar categorías
    experiencias_validas = ['Principiante', 'Intermedio', 'Avanzado']
    if 'experiencia' in usuario_info and usuario_info['experiencia'] not in experiencias_validas:
        errores.append(f"Experiencia debe ser una de: {experiencias_validas}")
    
    objetivos_validos = ['Pérdida de peso', 'Ganar músculo', 'Resistencia', 'Salud general']
    if 'objetivo' in usuario_info and usuario_info['objetivo'] not in objetivos_validos:
        errores.append(f"Objetivo debe ser uno de: {objetivos_validos}")
    
    equipos_validos = ['Gimnasio completo', 'Casa con equipo', 'Solo peso corporal']
    if 'equipo' in usuario_info and usuario_info['equipo'] not in equipos_validos:
        errores.append(f"Equipo debe ser uno de: {equipos_validos}")
    
    return (len(errores) == 0, errores)


def formatear_rutina_markdown(rutina_texto: str) -> str:
    """
    Formatea el texto de la rutina generada por el LLM para mejor visualización.
    
    Args:
        rutina_texto: Texto de la rutina generado por el LLM
        
    Returns:
        Texto formateado en Markdown
    """
    # Agregar título principal si no existe
    if not rutina_texto.startswith('#'):
        rutina_texto = f"# 🏋️ Tu Rutina Personalizada\n\n{rutina_texto}"
    
    return rutina_texto


def calcular_estadisticas_cluster(df: pd.DataFrame, cluster_id: int) -> Dict:
    """
    Calcula estadísticas detalladas de un cluster.
    
    Args:
        df: DataFrame con usuarios y clusters
        cluster_id: ID del cluster
        
    Returns:
        Diccionario con estadísticas del cluster
    """
    cluster_data = df[df['cluster'] == cluster_id]
    
    estadisticas = {
        'n_usuarios': len(cluster_data),
        'porcentaje_total': len(cluster_data) / len(df) * 100,
        
        # Estadísticas numéricas
        'tiempo': {
            'media': cluster_data['tiempo_disponible'].mean(),
            'mediana': cluster_data['tiempo_disponible'].median(),
            'std': cluster_data['tiempo_disponible'].std(),
            'min': cluster_data['tiempo_disponible'].min(),
            'max': cluster_data['tiempo_disponible'].max()
        },
        'frecuencia': {
            'media': cluster_data['frecuencia_semanal'].mean(),
            'mediana': cluster_data['frecuencia_semanal'].median(),
            'std': cluster_data['frecuencia_semanal'].std()
        },
        'edad': {
            'media': cluster_data['edad'].mean(),
            'mediana': cluster_data['edad'].median(),
            'std': cluster_data['edad'].std()
        },
        
        # Distribuciones categóricas
        'experiencia_dist': cluster_data['experiencia'].value_counts(normalize=True).to_dict(),
        'objetivo_dist': cluster_data['objetivo'].value_counts(normalize=True).to_dict(),
        'equipo_dist': cluster_data['equipo'].value_counts(normalize=True).to_dict(),
        'restricciones_dist': cluster_data['restricciones'].value_counts(normalize=True).to_dict()
    }
    
    return estadisticas


def generar_recomendaciones_adicionales(usuario_info: Dict, cluster_profile: Dict) -> List[str]:
    """
    Genera recomendaciones adicionales basadas en el perfil del usuario.
    
    Args:
        usuario_info: Información del usuario
        cluster_profile: Perfil del cluster
        
    Returns:
        Lista de recomendaciones
    """
    recomendaciones = []
    
    # Recomendaciones basadas en experiencia
    if usuario_info['experiencia'] == 'Principiante':
        recomendaciones.append("🔰 Como principiante, enfócate en la técnica correcta antes que en el peso")
        recomendaciones.append("📱 Considera usar videos de referencia para aprender los ejercicios")
    
    # Recomendaciones basadas en tiempo
    if usuario_info['tiempo_disponible'] < 30:
        recomendaciones.append("⏱️ Con sesiones cortas, los ejercicios compuestos son tu mejor aliado")
        recomendaciones.append("🔥 Considera entrenamientos HIIT para maximizar resultados en poco tiempo")
    
    # Recomendaciones basadas en objetivo
    if usuario_info['objetivo'] == 'Pérdida de peso':
        recomendaciones.append("🥗 La nutrición es crucial: mantén un déficit calórico moderado")
        recomendaciones.append("🚶 Considera agregar caminatas diarias de 20-30 minutos")
    elif usuario_info['objetivo'] == 'Ganar músculo':
        recomendaciones.append("💪 Asegura consumir suficiente proteína: 1.6-2.2g por kg de peso corporal")
        recomendaciones.append("😴 El descanso es crucial para el crecimiento muscular")
    
    # Recomendaciones basadas en frecuencia
    if usuario_info['frecuencia_semanal'] >= 5:
        recomendaciones.append("🔄 Con alta frecuencia, alterna entre grupos musculares para evitar sobreentrenamiento")
    
    # Recomendaciones basadas en restricciones
    if usuario_info['restricciones'] != 'Ninguna':
        recomendaciones.append(f"⚠️ Ten especial cuidado con ejercicios que involucren {usuario_info['restricciones'].lower()}")
        recomendaciones.append("🏥 Consulta con un fisioterapeuta si el dolor persiste o empeora")
    
    return recomendaciones
