import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

class CloudSegmentation:
    def __init__(self, eps=0.2, min_points=100):
        """
        Inicializa el segmentador con parámetros para DBSCAN
        
        Args:
            eps (float): Radio de búsqueda para vecinos en DBSCAN
            min_points (int): Número mínimo de puntos para formar un cluster
        """
        self.eps = eps
        self.min_points = min_points
        self.colores = None
        self.labels = None
        
    def cargar_nube(self, archivo):
        """
        Carga una nube de puntos desde un archivo .bin o .ply
        """
        extension = archivo.lower().split('.')[-1]
        
        if extension == 'bin':
            nube_puntos = np.fromfile(archivo, dtype=np.float32)
            nube_puntos = nube_puntos.reshape((-1, 4))
            puntos_xyz = nube_puntos[:, :3]
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(puntos_xyz)
            
        elif extension == 'ply':
            pcd = o3d.io.read_point_cloud(archivo)
            
        else:
            raise ValueError(f"Formato de archivo no soportado: {extension}. Use .bin o .ply")
            
        return pcd

    def segment_clusters(self, pcd, custom_colors=None):
        """
        Segmenta la nube de puntos usando DBSCAN y asigna colores a los clusters
        
        Args:
            pcd (open3d.geometry.PointCloud): Nube de puntos de entrada
            custom_colors (dict): Diccionario opcional para asignar colores específicos a clusters
        """
        # Calcular normales si no existen
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50)
            )
            pcd.orient_normals_towards_camera_location([0, 0, 0])

        # Realizar clustering DBSCAN
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            self.labels = np.array(
                pcd.cluster_dbscan(eps=self.eps, 
                                 min_points=self.min_points, 
                                 print_progress=True))

        max_label = self.labels.max()
        print(f"Se encontraron {max_label + 1} clusters")

        # Asignar colores basados en características de los clusters
        points = np.asarray(pcd.points)
        colors = np.zeros((len(points), 3))
        
        for cluster_id in range(max_label + 1):
            # Obtener puntos del cluster actual
            cluster_points = points[self.labels == cluster_id]
            
            # Analizar características del cluster
            cluster_height = np.mean(cluster_points[:, 2])  # Altura media
            cluster_size = len(cluster_points)  # Tamaño del cluster
            cluster_spread = np.std(cluster_points, axis=0)  # Dispersión
            
            # Clasificar clusters basado en características
            if cluster_height < 0.3:  # Cerca del suelo
                colors[self.labels == cluster_id] = [0.8, 0.2, 0.2]  # Rojo (suelo)
            elif cluster_spread[2] > 2.0:  # Alta variación vertical
                colors[self.labels == cluster_id] = [0.2, 0.8, 0.2]  # Verde (árboles/vegetación)
            elif cluster_spread[2] < 0.5 and cluster_size > 100:  # Poca variación vertical y grande
                colors[self.labels == cluster_id] = [0.2, 0.2, 0.8]  # Azul (edificios/estructuras)
            else:
                colors[self.labels == cluster_id] = [0.7, 0.7, 0.7]  # Gris (otros objetos)
        
        # Marcar puntos de ruido en negro
        colors[self.labels < 0] = [0.1, 0.1, 0.1]
        
        # Actualizar colores en la nube de puntos
        self.colores = colors
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

    def get_colored_cloud(self):
        """
        Retorna los colores asignados a la nube de puntos
        """
        return self.colores

    def get_labels(self):
        """
        Retorna las etiquetas de los clusters
        """
        return self.labels