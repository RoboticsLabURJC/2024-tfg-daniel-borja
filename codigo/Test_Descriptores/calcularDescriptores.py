import numpy as np
import open3d as o3d
from typing import Union, List

class LidarPointCloudDescriptors:

    def __init__(self, point_cloud: Union[str, o3d.geometry.PointCloud]):
        """
        Inicializa la clase con una nube de puntos.
        
        :param point_cloud: Ruta al archivo .ply o .bin, o un objeto PointCloud de Open3D
        """
        if isinstance(point_cloud, str):
            # Cargar nube de puntos desde archivo
            if point_cloud.endswith('.ply'):
                self.point_cloud = o3d.io.read_point_cloud(point_cloud)
            elif point_cloud.endswith('.bin'):
                # Asumiendo datos binarios en formato float32
                points = np.fromfile(point_cloud, dtype=np.float32)
                points = points.reshape([-1, 3])  # Reshape a coordenadas x, y, z
                self.point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            else:
                raise ValueError("Formato de archivo no soportado. Use .ply o .bin")
        elif isinstance(point_cloud, o3d.geometry.PointCloud):
            self.point_cloud = point_cloud
        else:
            raise TypeError("Debe proporcionar una ruta de archivo o un objeto PointCloud")
        
        # Calcular normales si no están ya calculadas
        if not self.point_cloud.has_normals():
            self.point_cloud.estimate_normals()
            self.point_cloud.orient_normals_consistent_tangent_plane(100)
    
    def calcular_altura_relativa(self, k_vecinos) -> np.ndarray:
        """
        Calcula la altura relativa para cada punto usando su vecindad local.

        :param k_vecinos: Número de vecinos para el análisis local
        :return: Arreglo con los valores de altura relativa por punto
        """
        point_cloud_tree = o3d.geometry.KDTreeFlann(self.point_cloud)
        puntos = np.asarray(self.point_cloud.points)
        alturas_relativas = np.zeros(len(puntos))

        for i, punto in enumerate(puntos):
            [k, idx, _] = point_cloud_tree.search_knn_vector_3d(punto, k_vecinos)
            vecinos = puntos[idx[1:]]  # Excluir el punto actual

            # Determinar la altura mínima en el vecindario de este punto
            altura_minima_vecinos = np.min(vecinos[:, 2])

            # Calcular la altura relativa como la diferencia entre la altura del punto y la altura mínima
            altura_relativa = punto[2] - altura_minima_vecinos
            alturas_relativas[i] = altura_relativa

        return alturas_relativas

    
    def calcular_planaridad(self, k_vecinos):
        """
        Calcula la planaridad de cada punto en la nube de puntos.

        :param k_vecinos: Número de vecinos a considerar para calcular la planaridad
        :return: Arreglo de planaridades
        """
        planaridades = np.zeros(len(self.point_cloud.points))
        kdtree = o3d.geometry.KDTreeFlann(self.point_cloud)

        for i in range(len(self.point_cloud.points)):
            # Buscar vecinos más cercanos
            [_, idx, _] = kdtree.search_knn_vector_3d(self.point_cloud.points[i], k_vecinos)
            vecinos = np.asarray(self.point_cloud.points)[idx, :]

            # Calcular matriz de covarianza
            cov = np.cov(vecinos.T)

            # Calcular valores propios
            valores_propios, _ = np.linalg.eigh(cov)
            valores_propios = np.sort(valores_propios)

            # Calcular planaridad (evitar división por cero)
            if valores_propios[2] > 0:
                planaridades[i] = (valores_propios[1] - valores_propios[0]) / valores_propios[2]
            else:
                planaridades[i] = 0.0  # Asignar 0 si no hay suficiente información

        return planaridades

    
    def calcular_anisotropia(self, k_vecinos) -> np.ndarray:
        """
        Calcula la anisotropía de cada punto usando análisis de componentes principales.
        
        :param k_vecinos: Número de vecinos para el análisis local
        :return: Arreglo con los valores de anisotropía por punto
        """
        point_cloud_tree = o3d.geometry.KDTreeFlann(self.point_cloud)
        puntos = np.asarray(self.point_cloud.points)
        anisotropias = np.zeros(len(puntos))

        for i, punto in enumerate(puntos):
            [k, idx, _] = point_cloud_tree.search_knn_vector_3d(punto, k_vecinos)
            vecinos = puntos[idx[1:]]  # Excluir el punto actual

            if len(vecinos) < 3:
                anisotropias[i] = 0
                continue

            # Análisis de componentes principales
            matriz_covarianza = np.cov(vecinos.T)
            valores_propios, _ = np.linalg.eig(matriz_covarianza)
            valores_propios = np.sort(valores_propios)

            # Calcular anisotropía con manejo de división por cero
            if valores_propios[2] != 0:
                anisotropia = (valores_propios[2] - valores_propios[1]) / valores_propios[2]
            else:
                anisotropia = 0

            anisotropias[i] = anisotropia

        return anisotropias

    
    def calcular_orientacion_normal(self) -> np.ndarray:
        """
        Calcula la orientación de las normales para cada punto.
        
        :return: Lista de orientaciones promedio para cada punto
        """
        normales = np.asarray(self.point_cloud.normals)
        return normales  # Ya que cada punto tiene su propia normal
    
    
    def calcular_variacion_superficial(self, k_vecinos):
        """
        Calcula la variación superficial de cada punto en la nube.
        
        Parámetros:
        -----------
        k_vecinos : int, opcional (por defecto=30)
            Número de vecinos más cercanos a considerar para cada punto
        
        Retorna:
        --------
        numpy.ndarray
            Array de variaciones superficiales para cada punto
        """
        # Convertir la nube de puntos a array de numpy
        puntos = np.asarray(self.point_cloud.points)
        
        # Crear el árbol KD para búsqueda eficiente de vecinos
        point_cloud_tree = o3d.geometry.KDTreeFlann(self.point_cloud)
        
        # Array para almacenar variaciones superficiales
        variaciones_superficiales = np.zeros(len(puntos))
        
        # Calcular variación superficial para cada punto
        for i, punto in enumerate(puntos):
            # Buscar los k_vecinos más cercanos
            [k, idx, _] = point_cloud_tree.search_knn_vector_3d(punto, k_vecinos)
            
            # Obtener los vecinos del punto actual
            vecinos = puntos[idx[1:]]  # Excluir el punto actual
            
            # Calcular la matriz de covarianza de los vecinos
            if len(vecinos) > 3:
                matriz_covarianza = np.cov(vecinos.T)
                
                # Calcular valores propios de la matriz de covarianza
                valores_propios, _ = np.linalg.eig(matriz_covarianza)
                
                # Ordenar valores propios de menor a mayor
                valores_propios_ordenados = np.sort(valores_propios)
                
                # Calcular variación superficial
                # λ1 (menor) corresponde a la dirección normal a la superficie
                # λ2 y λ3 corresponden a las direcciones del plano de la superficie
                if np.sum(valores_propios_ordenados) > 0:
                    # Variación como proporción del valor propio más pequeño
                    variacion = valores_propios_ordenados[0] / np.sum(valores_propios_ordenados)
                    variaciones_superficiales[i] = variacion
        
        return variaciones_superficiales

    
    def calcular_descriptores(self) -> dict:
        """
        Calcula todos los descriptores de la nube de puntos.
        
        :return: Diccionario con todos los descriptores
        """
        
        return {
            'altura_relativa': self.calcular_altura_relativa(),
            #'planaridad': self.calcular_planaridad(),
            #'anisotropia': self.calcular_anisotropia(),
            'orientacion_normal': self.calcular_orientacion_normal()
        }