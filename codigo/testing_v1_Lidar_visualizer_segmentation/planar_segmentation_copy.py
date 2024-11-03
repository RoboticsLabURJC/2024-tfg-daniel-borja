import open3d as o3d
import numpy as np

class GroundSegmentation:
    def __init__(self,  altura_max_suelo=0.5, distancia_threshold=0.1):
        self.altura_max_suelo = altura_max_suelo
        self.distancia_threshold = distancia_threshold
        self.colores = None

        # self.ground_cloud = None
        # self.non_ground_cloud = None

    def cargar_nube(self, archivo):
        """
        Carga una nube de puntos desde un archivo .bin o .ply
        
        Args:
            archivo (str): Ruta al archivo (.bin o .ply)
            
        Returns:
            open3d.geometry.PointCloud: Nube de puntos cargada
        """
        extension = archivo.lower().split('.')[-1]
        
        if extension == 'bin':
            # Cargar archivo .bin
            nube_puntos = np.fromfile(archivo, dtype=np.float32)
            nube_puntos = nube_puntos.reshape((-1, 4))
            puntos_xyz = nube_puntos[:, :3]
            
            # Crear nube de puntos Open3D
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(puntos_xyz)
            
        elif extension == 'ply':
            # Cargar archivo .ply directamente con Open3D
            pcd = o3d.io.read_point_cloud(archivo)
            
        else:
            raise ValueError(f"Formato de archivo no soportado: {extension}. Use .bin o .ply")
            
        return pcd

    def segment_ground(self, pcd):
        """
        Segmenta el suelo de una nube de puntos y cambia su color.
        
        Args:
            pcd (open3d.geometry.PointCloud): Nube de puntos de entrada
        """
        # 1. Filtro por altura
        puntos = np.asarray(pcd.points)
        mascara_altura = puntos[:, 2] < self.altura_max_suelo
        
        # Crear nube filtrada por altura
        pcd_filtrado = o3d.geometry.PointCloud()
        pcd_filtrado.points = o3d.utility.Vector3dVector(puntos[mascara_altura])
        
        # 2. RANSAC para encontrar el plano del suelo
        plane_model, inliers = pcd_filtrado.segment_plane(
            distance_threshold=self.distancia_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        # Verificar que el plano es aproximadamente horizontal
        [a, b, c, d] = plane_model
        angulo_normal = np.abs(np.arccos(c) * 180 / np.pi)
        if not (80 <= angulo_normal <= 100):
            print("Advertencia: El plano detectado no parece ser horizontal")
        
        # Obtener índices originales
        indices_originales = np.where(mascara_altura)[0][inliers]

        self.colores = np.zeros((len(puntos),3))
        
        # Colorear las nubes
        self.colores[indices_originales] = [1, 0, 0]  # Rojo para el suelo
        self.colores[~mascara_altura] = [0, 0.7, 0]  # Verde para no suelo
        
        pcd.colors = o3d.utility.Vector3dVector(self.colores)
        
    def get_colored_cloud(self):
        """
        Retorna los colores asignados a la nube de puntos.
        
        Returns:
            numpy.ndarray: Array con los colores RGB asignados
        """
        return self.colores
