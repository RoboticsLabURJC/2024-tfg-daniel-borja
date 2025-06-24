import open3d as o3d
import numpy as np
import time

class GroundSegmentation:

    def __init__(self, distancia_threshold=0.1, max_angle_deg=40.0, min_inliers=500):
        self.distancia_threshold = distancia_threshold
        self.max_angle_deg = max_angle_deg
        self.min_inliers = min_inliers
        self.distancia_threshold = distancia_threshold
        self.colores = None

        self.all_ground_mask = []
        self.all_semantic_labels = []
        self.global_iou_calculated = False
        self.segmentation_times = []

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
        Segmenta el suelo detectando múltiples planos válidos.
        """
        start_time = time.time()
        puntos = np.asarray(pcd.points)
        
        # --- Filtrado por altura con percentiles amplios ---
        z_values = puntos[:, 2]
        z_min = np.percentile(z_values, 0)
        z_max = np.percentile(z_values, 40)
        pcd_filtrado, mask = self.filtrar_por_altura(pcd, z_min=z_min, z_max=z_max)
        indices_filtrados = np.where(mask)[0]

        all_inliers_original = []

        # Crear copia para no modificar original
        restante = pcd_filtrado

        for i in range(25):  # Más iteraciones = más planos
            if len(restante.points) < 3:
                break  # No quedan puntos suficientes

            plane_model, inliers = restante.segment_plane(
                distance_threshold=self.distancia_threshold,
                ransac_n=3,
                num_iterations=1500
            )

            if len(inliers) == 0:
                continue

            a, b, c, d = plane_model
            angulo = np.rad2deg(np.arccos(abs(c)))

            if angulo < self.max_angle_deg and len(inliers) > self.min_inliers:
                print(f"[Plano {i+1}] Ángulo: {round(angulo, 2)}° | Inliers: {len(inliers)} ✅")
                # Mapear inliers al índice original
                inliers_global = indices_filtrados[np.asarray(inliers)]
                all_inliers_original.extend(inliers_global)

                # Quitar puntos usados para encontrar otros planos
                restante = restante.select_by_index(inliers, invert=True)
                indices_filtrados = np.delete(indices_filtrados, inliers)
            else:
                print(f"[Plano {i+1}] Ángulo: {round(angulo, 2)}° | Inliers: {len(inliers)} ❌")

        if not all_inliers_original:
            print("❌ No se detectaron planos válidos.")
            colores = np.zeros((len(puntos), 3))
            colores[:] = [0, 0.7, 0]
            self.colores = colores
            pcd.colors = o3d.utility.Vector3dVector(colores)
            return

        all_inliers_original = np.array(list(set(all_inliers_original)))  # Quitar duplicados

        # Colorear: rojo = suelo, verde = resto
        colores = np.zeros((len(puntos), 3))
        colores[all_inliers_original] = [1, 0, 0]  # Rojo
        resto = list(set(range(len(puntos))) - set(all_inliers_original))
        colores[resto] = [0, 1, 0]  # Verde

        self.colores = colores
        pcd.colors = o3d.utility.Vector3dVector(colores)

        elapsed = time.time() - start_time
        self.segmentation_times.append(elapsed)
        print(f"✅ Total puntos clasificados como suelo: {len(all_inliers_original)}")
        print(f"⏱️ Tiempo: {elapsed:.3f} s")

    def get_ground_mask(self, pcd):
        """
        Devuelve una máscara booleana donde True indica que el punto fue clasificado como suelo.
        """
        if self.colores is None:
            print("⚠️ No se ha ejecutado segment_ground aún.")
            return np.zeros(len(pcd.points), dtype=bool)
        
        colores = self.colores
        # Rojo → suelo
        return np.all(colores == [1, 0, 0], axis=1)
    
    def evaluate_segmentation_with_iou(self, pcd, semantic_labels):
        """
        Compara la segmentación geométrica vs etiquetas semánticas usando IoU.
        
        Parámetros:
        - pcd: open3d.geometry.PointCloud con los puntos de la escena
        - semantic_labels: np.ndarray con etiquetas ya mapeadas a [0=Drivable, 1=No Drivable]
        """
        puntos = np.asarray(pcd.points)

        # Obtener máscara de suelo según segmentación geométrica
        if self.colores is None or len(self.colores) != len(puntos):
            print("❌ Error: No se encontró segmentación válida. Ejecuta segment_ground() antes.")
            return

        ground_mask = np.all(self.colores == [1, 0, 0], axis=1)  # Rojo = suelo

        if len(ground_mask) != len(semantic_labels):
            print(f"❌ Error: Longitudes distintas: {len(ground_mask)} vs {len(semantic_labels)}")
            return
        
        # Acumular resultados para cálculo global
        self.all_ground_mask.append(ground_mask)
        self.all_semantic_labels.append(semantic_labels)



        # Suelo semántico = etiquetas == 0
        semantic_ground = (semantic_labels == 0)
        semantic_nonground = (semantic_labels == 1)
        geom_nonground = ~ground_mask

        # IoU para suelo
        inter_ground = np.logical_and(ground_mask, semantic_ground).sum()
        union_ground = np.logical_or(ground_mask, semantic_ground).sum()
        iou_ground = inter_ground / union_ground if union_ground > 0 else 0

        # IoU para no-suelo
        inter_nonground = np.logical_and(geom_nonground, semantic_nonground).sum()
        union_nonground = np.logical_or(geom_nonground, semantic_nonground).sum()
        iou_nonground = inter_nonground / union_nonground if union_nonground > 0 else 0

        print(f"\n📊 Evaluación de Segmentación:")
        print(f"✅ IoU Suelo (Drivable):        {iou_ground:.3f}")
        print(f"✅ IoU No Suelo (No Drivable):  {iou_nonground:.3f}")

    def calculate_global_iou(self):
        """
        Calcula el IoU global considerando todos los puntos acumulados.
        """
        if not self.all_ground_mask or not self.all_semantic_labels:
            print("⚠️ No hay datos acumulados para calcular IoU global.")
            return

        # Concatenar todos los resultados acumulados
        all_ground_mask = np.concatenate(self.all_ground_mask)
        all_semantic_labels = np.concatenate(self.all_semantic_labels)

        semantic_ground = (all_semantic_labels == 0)
        semantic_nonground = (all_semantic_labels == 1)
        geom_nonground = ~all_ground_mask

        # Cálculo global
        inter_ground = np.logical_and(all_ground_mask, semantic_ground).sum()
        union_ground = np.logical_or(all_ground_mask, semantic_ground).sum()
        iou_ground = inter_ground / union_ground if union_ground > 0 else 0

        inter_nonground = np.logical_and(geom_nonground, semantic_nonground).sum()
        union_nonground = np.logical_or(geom_nonground, semantic_nonground).sum()
        iou_nonground = inter_nonground / union_nonground if union_nonground > 0 else 0
        
        if self.segmentation_times:
            avg_time = sum(self.segmentation_times) / len(self.segmentation_times)
            print(f"\n📊 Evaluación de Segmentación GLOBAL:")
            print(f"✅ IoU Suelo (Drivable):        {iou_ground:.3f}")
            print(f"✅ IoU No Suelo (No Drivable):  {iou_nonground:.3f}")
            print(f"⏱️  Tiempo medio de segmentación: {avg_time:.3f} segundos/nube")
            print(f"📌 Total de puntos considerados: {len(all_ground_mask):,}")
        else:
            print("\n⚠️ No hay datos de tiempos de segmentación registrados.")
            
        self.global_iou_calculated = True



    def get_colored_cloud(self):
        return self.colores
    