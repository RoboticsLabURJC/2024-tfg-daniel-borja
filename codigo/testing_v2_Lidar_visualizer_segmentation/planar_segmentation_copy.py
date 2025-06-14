import open3d as o3d
import numpy as np

class GroundSegmentation:

    def __init__(self, distancia_threshold=0.3, max_angle_deg=3.0, min_inliers=10000):
        self.distancia_threshold = distancia_threshold
        self.max_angle_deg = max_angle_deg
        self.min_inliers = min_inliers
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
    
    def filtrar_por_altura(self, pcd, z_min=None, z_max=None):
        """
        Filtra la nube para conservar solo puntos con coordenada z entre z_min y z_max.
        Si alguno es None, no se aplica ese límite.
        """
        puntos = np.asarray(pcd.points)
        mask = np.ones(len(puntos), dtype=bool)
        if z_min is not None:
            mask = mask & (puntos[:, 2] >= z_min)
        if z_max is not None:
            mask = mask & (puntos[:, 2] <= z_max)
        
        puntos_filtrados = puntos[mask]
        pcd_filtrado = o3d.geometry.PointCloud()
        pcd_filtrado.points = o3d.utility.Vector3dVector(puntos_filtrados)
        return pcd_filtrado, mask
    
    def segment_ground(self, pcd):
        """
        Segmenta el suelo de una nube de puntos usando múltiples RANSAC
        y seleccionando el plano más horizontal (cercano a normal Z).

        Args:
            pcd (open3d.geometry.PointCloud): Nube de puntos de entrada
        """
        puntos = np.asarray(pcd.points)

        # Filtrado por altura (ajusta los límites según tus datos)
        pcd_filtrado, mask = self.filtrar_por_altura(pcd, z_min=-4.5, z_max=-0.5)
        indices_filtrados = np.where(mask)[0]

        mejor_inliers = []
        mejor_plano = None
        mejor_angulo = 999

        for i in range(10):  # Más repeticiones para mejorar robustez frente a ruido
            plane_model, inliers = pcd_filtrado.segment_plane(
                distance_threshold=self.distancia_threshold,
                ransac_n=3,
                num_iterations=1500
            )
            a, b, c, d = plane_model
            angulo = np.abs(np.arccos(c) * 180 / np.pi)
            diferencia_con_horizontal = abs(90 - angulo)

            print(f"[Iteración {i+1}] Ángulo con eje Z: {round(angulo, 2)}° | Inliers: {len(inliers)}")

            if angulo < self.max_angle_deg and len(inliers) > self.min_inliers:
                    if angulo < mejor_angulo:
                        mejor_angulo = angulo
                        mejor_plano = plane_model
                        mejor_inliers = inliers
                        print("↪️  -> Seleccionado como mejor plano actual.")

        if mejor_plano is None:
            print("❌ No se detectó un plano válido como suelo.")
            return
        
        # Mapeo de índices inliers a la nube original
        mejor_inliers_original = indices_filtrados[mejor_inliers]

        print(f"✅ Suelo detectado con ángulo Z: {90 - mejor_angulo:.2f}° respecto a horizontal. "
            f"{len(mejor_inliers)} puntos detectados como suelo.")

        # Colores: rojo = suelo, verde = resto
        colores = np.zeros((len(puntos), 3))
        colores[mejor_inliers_original] = [1, 0, 0]  # rojo para suelo
        resto = list(set(range(len(puntos))) - set(mejor_inliers_original))
        colores[resto] = [0, 0.7, 0]  # verde para resto

        self.colores = colores
        pcd.colors = o3d.utility.Vector3dVector(colores)

    def get_colored_cloud(self):
        return self.colores
    