"""
Sistema de Recomendación Smart Coach AI
Combina K-Means Clustering con LLM para generar rutinas personalizadas
"""
import pickle
import pandas as pd
from typing import Dict, Optional
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from utils import (
    crear_prompt_rutina,
    extraer_perfil_cluster,
    predecir_cluster,
    validar_input_usuario,
    formatear_rutina_markdown,
    generar_recomendaciones_adicionales
)


class SmartCoachRecommender:
    """
    Sistema de recomendación que combina clustering con LLM.
    """
    
    def __init__(self, 
                 modelo_path: str = 'kmeans_model.pkl',
                 scaler_path: str = 'scaler.pkl',
                 encoders_path: str = 'label_encoders.pkl',
                 dataset_path: str = 'usuarios_con_clusters.csv',
                 llm_model: str = 'llama3'):
        """
        Inicializa el sistema de recomendación.
        
        Args:
            modelo_path: Ruta al modelo KMeans guardado
            scaler_path: Ruta al scaler guardado
            encoders_path: Ruta a los label encoders
            dataset_path: Ruta al dataset con clusters
            llm_model: Nombre del modelo de Ollama a usar
        """
        # Cargar modelos y datos
        self.modelo_kmeans = self._cargar_pickle(modelo_path)
        self.scaler = self._cargar_pickle(scaler_path)
        self.label_encoders = self._cargar_pickle(encoders_path)
        self.df = pd.read_csv(dataset_path)
        
        # Inicializar LLM
        self.llm = Ollama(model=llm_model, temperature=0.7)
        
        print(f"✅ Smart Coach Recommender inicializado")
        print(f"   • Modelo: {llm_model}")
        print(f"   • Clusters disponibles: {len(self.df['cluster'].unique())}")
        print(f"   • Usuarios en dataset: {len(self.df)}")
    
    def _cargar_pickle(self, path: str):
        """Carga un archivo pickle."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo: {path}")
    
    def recomendar(self, usuario_info: Dict) -> Dict:
        """
        Genera una recomendación personalizada para un usuario.
        
        Args:
            usuario_info: Diccionario con información del usuario
            
        Returns:
            Diccionario con la rutina y metadata
        """
        # 1. Validar input
        es_valido, errores = validar_input_usuario(usuario_info)
        if not es_valido:
            return {
                'error': True,
                'mensaje': 'Error en la validación de datos',
                'errores': errores
            }
        
        try:
            # 2. Predecir cluster
            cluster_id = predecir_cluster(
                self.modelo_kmeans,
                self.scaler,
                self.label_encoders,
                usuario_info
            )
            
            # 3. Obtener perfil del cluster
            cluster_profile = extraer_perfil_cluster(self.df, cluster_id)
            
            # 4. Crear prompt personalizado
            prompt = crear_prompt_rutina(usuario_info, cluster_profile)
            
            # 5. Generar rutina con LLM
            print(f"🤖 Generando rutina con LLM para Cluster {cluster_id}...")
            rutina_texto = self.llm(prompt)
            
            # 6. Formatear respuesta
            rutina_formateada = formatear_rutina_markdown(rutina_texto)
            
            # 7. Generar recomendaciones adicionales
            recomendaciones = generar_recomendaciones_adicionales(
                usuario_info, 
                cluster_profile
            )
            
            # 8. Compilar resultado
            resultado = {
                'error': False,
                'rutina': rutina_formateada,
                'cluster_id': int(cluster_id),
                'cluster_profile': cluster_profile,
                'recomendaciones_adicionales': recomendaciones,
                'metadata': {
                    'usuario': usuario_info,
                    'n_usuarios_similares': cluster_profile['n_usuarios']
                }
            }
            
            return resultado
            
        except Exception as e:
            return {
                'error': True,
                'mensaje': f'Error al generar recomendación: {str(e)}',
                'errores': [str(e)]
            }
    
    def obtener_info_cluster(self, cluster_id: int) -> Optional[Dict]:
        """
        Obtiene información detallada de un cluster específico.
        
        Args:
            cluster_id: ID del cluster
            
        Returns:
            Diccionario con información del cluster
        """
        if cluster_id not in self.df['cluster'].unique():
            return None
        
        return extraer_perfil_cluster(self.df, cluster_id)
    
    def listar_clusters(self) -> pd.DataFrame:
        """
        Lista todos los clusters con sus características principales.
        
        Returns:
            DataFrame con resumen de clusters
        """
        clusters_info = []
        
        for cluster_id in sorted(self.df['cluster'].unique()):
            perfil = extraer_perfil_cluster(self.df, cluster_id)
            clusters_info.append({
                'Cluster': cluster_id,
                'N° Usuarios': perfil['n_usuarios'],
                'Experiencia': perfil['experiencia_comun'],
                'Objetivo': perfil['objetivo_comun'],
                'Equipo': perfil['equipo_comun'],
                'Tiempo Prom.': f"{perfil['tiempo_promedio']:.0f} min",
                'Frecuencia': f"{perfil['frecuencia_promedio']:.1f} días",
                'Edad Prom.': f"{perfil['edad_promedio']:.0f} años"
            })
        
        return pd.DataFrame(clusters_info)


def ejemplo_uso():
    """
    Función de ejemplo para demostrar el uso del sistema.
    """
    print("=" * 80)
    print("🏋️ SMART COACH AI - Ejemplo de Uso")
    print("=" * 80)
    
    # Inicializar sistema
    coach = SmartCoachRecommender()
    
    # Ejemplo de usuario
    usuario = {
        'experiencia': 'Intermedio',
        'objetivo': 'Ganar músculo',
        'equipo': 'Gimnasio completo',
        'tiempo_disponible': 60,
        'frecuencia_semanal': 4,
        'edad': 28,
        'restricciones': 'Ninguna'
    }
    
    print(f"\n📋 Perfil del Usuario:")
    for key, value in usuario.items():
        print(f"   • {key}: {value}")
    
    # Generar recomendación
    print(f"\n🔄 Generando recomendación personalizada...")
    resultado = coach.recomendar(usuario)
    
    if resultado['error']:
        print(f"\n❌ Error: {resultado['mensaje']}")
        for error in resultado['errores']:
            print(f"   • {error}")
    else:
        print(f"\n✅ Recomendación generada exitosamente!")
        print(f"\n📊 Cluster asignado: {resultado['cluster_id']}")
        print(f"👥 Usuarios similares en cluster: {resultado['cluster_profile']['n_usuarios']}")
        
        print(f"\n📝 RUTINA GENERADA:")
        print("=" * 80)
        print(resultado['rutina'])
        
        print(f"\n💡 RECOMENDACIONES ADICIONALES:")
        print("=" * 80)
        for rec in resultado['recomendaciones_adicionales']:
            print(f"  {rec}")
    
    # Mostrar información de clusters
    print(f"\n\n📊 CLUSTERS DISPONIBLES:")
    print("=" * 80)
    print(coach.listar_clusters().to_string(index=False))


if __name__ == "__main__":
    ejemplo_uso()
