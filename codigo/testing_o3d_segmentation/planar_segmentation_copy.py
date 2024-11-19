import open3d as o3d
import numpy as np

class GroundSegmentation:
    def __init__(self, altura_max_suelo=0.5, distancia_threshold=0.1):
        self.altura_max_suelo = altura_max_suelo
        self.distancia_threshold = distancia_threshold
        self.colores = None
        self.planes_colors = None

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
        Segmenta múltiples planos de una nube de puntos y asigna colores diferentes a cada plano.
        
        Args:
            pcd (open3d.geometry.PointCloud): Nube de puntos de entrada
        """
        # Calcular normales si no existen
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd.orient_normals_towards_camera_location([0, 0, 0])

        # Detectar planos usando detect_planar_patches
        planar_patches = pcd.detect_planar_patches(
            normal_variance_threshold_deg=80,  # Menor valor = planos mas precisos
            coplanarity_deg=60, # Mayor valor = distribucion mas ajustada depuntos
            outlier_ratio=0.75, 
            min_plane_edge_length=0.1,  # tamaño minimo de los planos
            min_num_points=100,  # Numero minimo de puntos por plano
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
        )

        # Generar colores aleatorios para cada plano
        np.random.seed(42)  # Para consistencia en los colores
        plane_colors = np.random.rand(len(planar_patches), 3)
        
        # Inicializar array de colores para todos los puntos
        points = np.asarray(pcd.points)
        self.colores = np.zeros((len(points), 3))
        points_assigned = np.zeros(len(points), dtype=bool)
        
        # Asignar colores a los puntos de cada plano
        for idx, plane in enumerate(planar_patches):
            # Obtener los puntos que pertenecen al plano actual
            plane_center = plane.center
            plane_normal = plane.R[:, 2]  # La tercera columna es la normal del plano
            
            # Calcular distancias de todos los puntos al plano
            point_to_plane = np.abs(np.dot(points - plane_center, plane_normal))
            plane_points = point_to_plane < self.distancia_threshold
            
            # Asignar color solo a puntos no asignados previamente
            new_points = plane_points & ~points_assigned
            self.colores[new_points] = plane_colors[idx]
            points_assigned[new_points] = True
        
        # Asignar color gris a puntos que no pertenecen a ningún plano
        self.colores[~points_assigned] = [0.5, 0.5, 0.5]
        
        # Actualizar colores en la nube de puntos
        pcd.colors = o3d.utility.Vector3dVector(self.colores)
import open3d as o3d
import numpy as np

class GroundSegmentation:
    def __init__(self, altura_max_suelo=1.0, distancia_threshold=0.2):
        self.altura_max_suelo = altura_max_suelo
        self.distancia_threshold = distancia_threshold
        self.colores = None
        self.planes_colors = None

    def cargar_nube(self, archivo):
        """
        Carga una nube de puntos desde un archivo .bin o .ply
        """
        # [Mantener el método igual que el original]
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
        Segmenta suelo y paredes en entornos naturales.
        """
        # Calcular normales con un radio mayor para superficies más irregulares
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.3,  # Aumentado para mejor estimación en superficies naturales
                    max_nn=50    # Más puntos para mejor estimación
                )
            )
            pcd.orient_normals_towards_camera_location([0, 0, 0])

        # Detectar planos con parámetros ajustados para entornos naturales
        planar_patches = pcd.detect_planar_patches(
            normal_variance_threshold_deg=70,    # Menor valor = planos más precisos
            coplanarity_deg=70,                 #  Mayor valor = distribución más ajustada de puntos
            outlier_ratio=0.6,                  # Mayor tolerancia a outliers
            min_plane_edge_length=0.2,          #  Tamaño mínimo de los planos
            min_num_points=50,                 # Número mínimo de puntos por plano
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)  # Más vecinos
        )

        # Generar colores específicos para suelo y paredes
        points = np.asarray(pcd.points)
        self.colores = np.zeros((len(points), 3))
        points_assigned = np.zeros(len(points), dtype=bool)
        
        # Colores predefinidos
        ground_color = np.array([0.8, 0.0, 0.0])  # Rojo para suelo
        wall_color = np.array([0.0, 0.7, 0.0])    # Verde para paredes
        
        for idx, plane in enumerate(planar_patches):
            plane_center = plane.center
            plane_normal = plane.R[:, 2]
            
            # Calcular ángulo con respecto a la vertical
            angle_with_vertical = np.arccos(np.abs(np.dot(plane_normal, [0, 0, 1]))) * 180 / np.pi
            
            # Calcular distancias de puntos al plano
            point_to_plane = np.abs(np.dot(points - plane_center, plane_normal))
            plane_points = point_to_plane < self.distancia_threshold
            
            # Clasificar como suelo o pared basado en el ángulo
            new_points = plane_points & ~points_assigned
            if angle_with_vertical < 30:  # Suelo (casi horizontal)
                self.colores[new_points] = ground_color
            elif angle_with_vertical > 60:  # Paredes (casi vertical)
                self.colores[new_points] = wall_color
            points_assigned[new_points] = True
        
        # Puntos no asignados en gris (vegetación/otros)
        self.colores[~points_assigned] = [0.7, 0.7, 0.7]
        
        # Actualizar colores en la nube de puntos
        pcd.colors = o3d.utility.Vector3dVector(self.colores)

    def get_colored_cloud(self):
        return self.colores
    def get_colored_cloud(self):
        """
        Retorna los colores asignados a la nube de puntos.
        
        Returns:
            numpy.ndarray: Array con los colores RGB asignados
        """
        return self.colores