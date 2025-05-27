import numpy as np
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
import sys
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm 
import time
from scipy.stats import truncnorm
import glob
from scipy.stats import mode
import traceback

# ------ Lectura de archivos -------

def read_pcd_file(file_path):
    """Lee archivos .pcd y devuelve los puntos e intensidades."""
    if not file_path.endswith('.pcd') or not os.path.exists(file_path):
        print(f"Error: .pcd file not found at {file_path}")
        sys.exit(1)
    pcd = o3d.io.read_point_cloud(file_path)  # Cargar archivo .pcd
    points = np.asarray(pcd.points)  # Extraer coordenadas XYZ
    
    # Verificar si el archivo tiene intensidades (remisiones)
    if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
        remissions = np.linalg.norm(np.asarray(pcd.colors), axis=1)  # Intensidad basada en colores
    else:
        remissions = np.zeros(len(points))  # Crear valores de intensidad predeterminados
    
    return points, remissions

def read_bin_file(file_path):
    """Lee archivos .bin y devuelve los puntos e intensidades."""
    if not file_path.endswith('.bin') or not os.path.exists(file_path):
        print(f"Error: .bin file not found at {file_path}")
        sys.exit(1)
    scan = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
    points, remissions = scan[:, 0:3], scan[:, 3]
    return points, remissions

def read_ply_file(file_path):
    """Lee archivos .ply y devuelve los puntos e intensidades."""
    if not file_path.endswith('.ply') or not os.path.exists(file_path):
        print(f"Error: .ply file not found at {file_path}")
        sys.exit(1)
    plydata = PlyData.read(file_path)
    x, y, z = plydata['vertex'].data['x'], plydata['vertex'].data['y'], plydata['vertex'].data['z']
    points = np.vstack((x, y, z)).T
    remissions = plydata['vertex'].data['intensity']
    return points, remissions

# ------ Densificación -------

def densify_point_cloud(points: np.ndarray, remissions: np.ndarray, labels: np.ndarray,
                        density_factor: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Densifica la nube de puntos con IDW y genera etiquetas para puntos nuevos.
    
    :param points: Nube de puntos original (N,3).
    :param remissions: Intensidades (N,).
    :param labels: Etiquetas originales (N,), opcional.
    :param density_factor: Factor de densificación.
    :return: points_combined, remissions_combined, labels_combined (si labels es None, retorna None).
    """

    print("\n=== INICIO DEL PROCESO DE DENSIFICACION MEDIANTE IDW CON 2 PUNTOS ===")
    print(f"Puntos iniciales: {points.shape}, Remisiones: {remissions.shape}")

    if labels is not None:
        print(f"Etiquetas iniciales: {labels.shape}")

    # Establecer el número de nuevos puntos a crear
    num_new_points = int((len(points) * (density_factor-1)))
    print(f"\n1. Generando {num_new_points} nuevos puntos...")
    # Seleccionar puntos base de manera balanceada
    indices = np.random.choice(len(points), size=num_new_points, replace=True)
    print("\nPuntos base seleccionados (índices):")
    print(indices[:10], "...")  # Mostramos solo los primeros 10 para no saturar

    #unique_indices, counts = np.unique(indices, return_counts=True)
    #point_usage_count = dict(zip(unique_indices, counts))
    #print("\nConteo de usos por punto (primeros 10):")
    #print(dict(zip(unique_indices[:10], counts[:10])), "...")
    
    # Encontrar vecinos más cercanos para todos los puntos seleccionados a la vez
    print("\n2. Buscando vecinos más cercanos...")
    nn = NearestNeighbors(n_neighbors=2).fit(points)
    distances, neighbor_indices = nn.kneighbors(points[indices])

    print("\nDistancias a vecinos (primeros 10):")
    print(distances[:10])
    print("\nÍndices de vecinos (primeros 10):")
    print(neighbor_indices[:10])

    # Los vecinos son la segunda columna (la primera es el punto mismo)
    neighbor_points = points[neighbor_indices[:, 1]]
    neighbor_remissions = remissions[neighbor_indices[:, 1]]
    print("\nPuntos vecinos (primeros 10):")
    print(neighbor_points[:10])

    #Parámetros de la distribución normal truncada
    print("\n3. Generando parámetros de interpolación...")
    media = 0.5
    desviacion_estandar = 0.2
    limite_inferior = 0
    limite_superior = 1   

    #Convertir los límites al espacio de la distribucion normal   
    a = (limite_inferior - media) / desviacion_estandar
    b = (limite_superior - media) / desviacion_estandar
    
    lambdas = truncnorm.rvs(a, b, loc=media, scale=desviacion_estandar, size=num_new_points)
    print("\nValores lambda generados (primeros 10):")
    print(lambdas[:10])

    # Interpolación vectorizada de puntos
    base_points = points[indices]
    interpolated_points = lambdas[:, None] * base_points + (1 - lambdas[:, None]) * neighbor_points
    print("\nPuntos interpolados (primeros 10):")
    print(interpolated_points[:10])

    # Interpolación vectorizada de remisiones
    d_A = np.linalg.norm(interpolated_points - base_points, axis=1)
    d_B = np.linalg.norm(interpolated_points - neighbor_points, axis=1)

    print("\nDistancias a puntos base (primeros 10):")
    print(d_A[:10])
    print("\nDistancias a vecinos (primeros 10):")
    print(d_B[:10])

    # Evitar divisiones por cero
    d_A = np.where(d_A == 0, 1e-10, d_A)
    d_B = np.where(d_B == 0, 1e-10, d_B)
    
    w_A = 1 / d_A
    w_B = 1 / d_B
    print("\nPesos para puntos base (primeros 10):")
    print(w_A[:10])
    print("\nPesos para vecinos (primeros 10):")
    print(w_B[:10])
    
    base_remissions = remissions[indices]
    interpolated_remissions = (w_A * base_remissions + w_B * neighbor_remissions) / (w_A + w_B)
    print("\nRemisiones base (primeros 10):")
    print(base_remissions[:10])
    print("\nRemisiones vecinas (primeros 10):")
    print(neighbor_remissions[:10])
    print("\nRemisiones interpoladas (primeros 10):")
    print(interpolated_remissions[:10])

    base_labels = labels[indices]
    neighbor_labels = labels[neighbor_indices[:, 1]]
    combined_labels_interpolated = np.array([mode([b, n], keepdims=True).mode[0] if b != n else b
                                             for b, n in zip(base_labels, neighbor_labels)])

    # 6. Combinar resultados
    print("\n6. Combinando resultados...")
    combined_points = np.vstack([points, interpolated_points])
    combined_remissions = np.concatenate([remissions, interpolated_remissions])
    combined_labels = np.concatenate([labels, combined_labels_interpolated])

    print("\n=== RESULTADOS FINALES ===")
    print(f"Total puntos originales: {len(points)}")
    print(f"Total puntos nuevos: {len(interpolated_points)}")
    print(f"Total puntos combinados: {len(combined_points)}")
    if combined_labels is not None:
        print(f"Total etiquetas combinadas: {len(combined_labels)}")
    #print("\nPrimeros 5 puntos originales:")
    #print(points[:5])
    #print("\nÚltimos 5 puntos generados:")
    #print(interpolated_points[-5:])

    return combined_points, combined_remissions, combined_labels 

# ------ Guardado de archivos-------

def save_point_cloud(points: np.ndarray, remissions: np.ndarray, output_path: str, input_extension: str):
    """
    Guarda la nube de puntos densificada en el formato especificado.
    
    :param points: Nube de puntos (N, 3).
    :param remissions: Intensidades de los puntos (N,).
    :param output_path: Ruta del archivo de salida.
    :param input_extension: Extensión del archivo de entrada (.bin, .ply, .pcd).
    """
    if input_extension == '.ply':
        # Guardar en formato PLY
        vertex = np.zeros(len(points), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
            ('intensity', 'f4')
        ])
        vertex['x'] = points[:, 0]
        vertex['y'] = points[:, 1]
        vertex['z'] = points[:, 2]
        vertex['intensity'] = remissions
        
        ply_element = PlyElement.describe(vertex, 'vertex')
        PlyData([ply_element]).write(output_path)
        print(f"Archivo PLY guardado en: {output_path}")
    
    elif input_extension == '.pcd':
        # Guardar en formato PCD usando Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if remissions is not None and len(remissions) > 0:
            # Normalizar intensidades a escala de 0-1 y asignarlas como colores
            colors = np.tile(remissions[:, None], (1, 3)) / np.max(remissions)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
        print(f"Archivo PCD guardado en: {output_path}")
    
    elif input_extension == '.bin':
        # Guardar en formato BIN
        combined = np.hstack((points, remissions[:, None]))
        combined.astype(np.float32).tofile(output_path)
        print(f"Archivo BIN guardado en: {output_path}")
    
    else:
        print(f"Formato '{input_extension}' no soportado. No se guardó el archivo.")

# ------ Procesamiento por lotes -------

def batch_densification(input_dir: str, density_factor: float = 5.0, label_dir: str = None):
    densified_folder_name = f"densified_IDW_2points_{density_factor}"
    output_dir = os.path.join(input_dir, densified_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    input_files = [f for f in os.listdir(input_dir) if f.endswith(('.bin', '.ply', '.pcd'))]
    if not input_files:
        print("\n⚠ No se encontraron archivos .bin, .ply o .pcd en el directorio")
        return

    print(f"\n📁 Archivos a procesar ({len(input_files)}):")
    for i, f in enumerate(input_files, 1):
        print(f" {i}. {f}")

    for file in tqdm(input_files, desc="Procesando archivos"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"densified_{file}")
        ext = os.path.splitext(file)[1]
        base_name = os.path.splitext(file)[0]

        try:
            # Leer puntos y remisiones según extensión
            if ext == '.bin':
                points, remissions = read_bin_file(input_path)
            elif ext == '.ply':
                points, remissions = read_ply_file(input_path)
            elif ext == '.pcd':
                points, remissions = read_pcd_file(input_path)

            labels = None
            if label_dir:
                # Mismo patrón que en tu ejemplo para buscar .label relacionado
                parts = base_name.split('_')
                if len(parts) > 1:
                    base_name_common = "_".join(parts[:-1])
                else:
                    base_name_common = base_name

                possible_labels = glob.glob(os.path.join(label_dir, f"*{base_name_common}*.label"))
                if len(possible_labels) == 0:
                    print(f"⚠ No se encontró archivo .label para {file} (buscando con patrón '{base_name_common}')")
                    labels = np.zeros(len(points), dtype=np.uint32)
                else:
                    label_path = possible_labels[0]
                    labels = np.fromfile(label_path, dtype=np.uint32)
                    print(f"labels shape: {labels.shape}, type: {type(labels)}")
                    print(f"labels sample: {labels[:10]}")
                    if len(labels) != len(points):
                        print(f"⚠ Mismatch: {file} tiene {len(points)} puntos, pero {len(labels)} etiquetas.")
                        labels = np.zeros(len(points), dtype=np.uint32)
            else:
                labels = np.zeros(len(points), dtype=np.uint32)

            print(f"\n🔍 Procesando: {file} ({len(points)} puntos)")

            # Adaptar densify_point_cloud para que reciba y devuelva etiquetas
            dense_points, dense_remissions, dense_labels = densify_point_cloud(
                points, remissions, labels, density_factor)

            print(f"✅ Densificado a {len(dense_points)} puntos")

            save_point_cloud(dense_points, dense_remissions, output_path, ext)

            # Guardar etiquetas densificadas
            label_output_path = os.path.splitext(output_path)[0] + ".label"
            dense_labels.astype(np.uint32).tofile(label_output_path)
            print(f"Archivo de etiquetas guardado en: {label_output_path}")

        except Exception as e:
            print(f"\n❌ Error procesando {file}: {str(e)}")
            print(f"❌ Error: {e}")
            traceback.print_exc()

    print("\n✅ Proceso de densificación completado.")
            

def main():
    print("\n" + "="*50)
    print(" DENSIFICADOR DE NUBES DE PUNTOS LIDAR CON ETIQUETAS")
    print("="*50 + "\n")

    while True:
        input_dir = input("📂 Introduzca la ruta del directorio de entrada: ").strip()
        if os.path.isdir(input_dir):
            break
        print("⚠ El directorio no existe. Intente nuevamente.")

    while True:
        label_dir = input("🏷️ Introduzca la ruta del directorio de etiquetas (.label) (opcional): ").strip()
        if label_dir == "":
            label_dir = None
            break
        elif os.path.isdir(label_dir):
            break
        print("⚠ El directorio no existe. Intente nuevamente.")

    while True:
        try:
            density_factor = float(input("\n🔢 Introduzca el factor de densificación (3-10 recomendado): "))
            if 1.0 <= density_factor <= 20.0:
                break
            print("⚠ El valor debe estar entre 1.0 y 20.0")
        except ValueError:
            print("⚠ Debe introducir un número válido")

    print(f"\n⚙ Configuración:")
    print(f" - Directorio de entrada: {input_dir}")
    print(f" - Directorio de etiquetas: {label_dir if label_dir else 'Ninguno'}")
    print(f" - Factor de densificación: {density_factor}")
    print(f" - Los resultados se guardarán en: {os.path.join(input_dir, f'densified_IDW_2points_{density_factor}')}")

    batch_densification(input_dir, density_factor, label_dir)


if __name__ == "__main__":
    main()