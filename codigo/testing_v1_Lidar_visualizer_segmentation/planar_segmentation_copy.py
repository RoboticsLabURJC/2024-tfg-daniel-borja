import open3d as o3d
import numpy as np

class GroundSegmentation:
    def __init__(self, altura_max_suelo=-3.0, distancia_threshold=5.0):

        self.altura_max_suelo = altura_max_suelo
        self.distancia_threshold = distancia_threshold
        self.colores = None

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

    def segment_ground(self, pcd):
        """
        Segmenta el suelo de una nube de puntos y cambia su color.
        """
        puntos = np.asarray(pcd.points)
        mascara_altura = puntos[:, 2] < self.altura_max_suelo
        
        pcd_filtrado = o3d.geometry.PointCloud()
        pcd_filtrado.points = o3d.utility.Vector3dVector(puntos[mascara_altura])
        
        plane_model, inliers = pcd_filtrado.segment_plane(
            distance_threshold=self.distancia_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        [a, b, c, d] = plane_model
        angulo_normal = np.abs(np.arccos(c) * 180 / np.pi)
        if not (80 <= angulo_normal <= 100):
            print("Advertencia: El plano detectado no parece ser horizontal")
        
        indices_originales = np.where(mascara_altura)[0][inliers]

        self.colores = np.zeros((len(puntos),3))
        self.colores[indices_originales] = [1, 0, 0]  # Rojo para el suelo
        self.colores[~mascara_altura] = [0, 0.7, 0]  # Verde para no suelo
        
        pcd.colors = o3d.utility.Vector3dVector(self.colores)
        
    def get_colored_cloud(self):
        return self.colores
    