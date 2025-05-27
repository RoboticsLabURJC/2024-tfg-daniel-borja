import numpy as np
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
import sys
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm 
import time
from scipy.stats import truncnorm, mode
import glob

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
    Densifica la nube de puntos utilizando interpolación basada en los dos vecinos más cercanos.
    """

    print("\n=== INICIO DEL PROCESO DE DENSIFICACION MEDIANTE IDW CON 3 PUNTOS ===")
    print(f"Puntos iniciales: {points.shape}, Remisiones: {remissions.shape}")

    #Parámetros de la distribución normal truncada
    media = 0.5
    desviacion_estandar = 0.2
    limite_inferior = 0
    limite_superior = 1   

    #Convertir los límites al espacio de la distribucion normal   
    a = (limite_inferior - media) / desviacion_estandar
    b = (limite_superior - media) / desviacion_estandar


    # Preparar vecinos más cercanos (3 vecinos)
    nn = NearestNeighbors(n_neighbors=3).fit(points)

    # Calcular número de nuevos puntos
    num_new_points = int((len(points) * (density_factor-1)))
    print(f"\n1. Generando {num_new_points} nuevos puntos...")

    
    # Seleccionar puntos base de manera balanceada
    indices = np.random.choice(len(points), size=num_new_points, replace=True)
    print("\nPuntos base seleccionados (primeros 10 índices):")
    print(indices[:10], "...")


    # Encontrar vecinos para todos los puntos seleccionados
    distances, neighbor_indices = nn.kneighbors(points[indices])

    # Preparar arrays para interpolación
    base_points = points[indices]
    neighbor_points_1 = points[neighbor_indices[:, 1]]
    neighbor_points_2 = points[neighbor_indices[:, 2]]

    # Generar lambdas para todos los puntos
    lambdas = truncnorm.rvs(a, b, loc=media, scale=desviacion_estandar, size=(num_new_points, 3))

    # Normalizar para que sumen 1
    lambdas = lambdas / lambdas.sum(axis=1, keepdims=True)

    # Interpolación vectorizada de puntos
    interpolated_points = (
        lambdas[:, 0, None] * base_points + 
        lambdas[:, 1, None] * neighbor_points_1 + 
        lambdas[:, 2, None] * neighbor_points_2
    )
    
    # Interpolación de remisiones
    d_A = np.linalg.norm(interpolated_points - base_points, axis=1)
    d_B = np.linalg.norm(interpolated_points - neighbor_points_1, axis=1)
    d_C = np.linalg.norm(interpolated_points - neighbor_points_2, axis=1)

    # Evitar divisiones por cero
    d_A = np.where(d_A == 0, 1e-10, d_A)
    d_B = np.where(d_B == 0, 1e-10, d_B)
    d_C = np.where(d_C == 0, 1e-10, d_C)

    w_A = 1 / d_A
    w_B = 1 / d_B
    w_C = 1 / d_C
    
    base_remissions = remissions[indices]
    neighbor_remissions_1 = remissions[neighbor_indices[:, 1]]
    neighbor_remissions_2 = remissions[neighbor_indices[:, 2]]
    
    interpolated_remissions = (
        (w_A * base_remissions + w_B * neighbor_remissions_1 + w_C * neighbor_remissions_2) / 
        (w_A + w_B + w_C)
    )

    base_labels = labels[indices]
    neighbor1_labels = labels[neighbor_indices[:, 1]]
    neighbor2_labels = labels[neighbor_indices[:, 2]]
    combined_labels_interpolated = np.array([mode([b, n1, n2], keepdims=True).mode[0]
                                             for b, n1, n2 in zip(base_labels, neighbor1_labels, neighbor2_labels)], dtype=labels.dtype)

    # Combinar resultados
    combined_points = np.vstack([points, interpolated_points])
    combined_remissions = np.concatenate([remissions, interpolated_remissions])
    combined_labels= np.concatenate([labels, combined_labels_interpolated])

    print("\n=== RESULTADOS FINALES ===")
    print(f"Total puntos originales: {len(points)}")
    print(f"Total puntos nuevos: {len(interpolated_points)}")
    print(f"Total puntos combinados: {len(combined_points)}")

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

def batch_densification(input_dir: str,
                        density_factor: float = 5.0,
                        label_dir: str = None):
    # Carpeta de salida
    densified_folder_name = f"densified_IDW_3points_{density_factor}"
    output_dir = os.path.join(input_dir, densified_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nLos archivos densificados se guardarán en: {output_dir}")

    # Listado de nubes
    input_files = [f for f in os.listdir(input_dir)
                   if f.endswith(('.bin', '.ply', '.pcd'))]
    if not input_files:
        print("\n⚠ No se encontraron archivos .bin, .ply o .pcd en el directorio")
        return

    print(f"\n📁 Archivos a procesar ({len(input_files)}):")
    for i, f in enumerate(input_files, 1):
        print(f" {i}. {f}")

    # Procesar cada archivo
    for file in tqdm(input_files, desc="Procesando archivos"):
        input_path = os.path.join(input_dir, file)
        base, ext = os.path.splitext(file)

        # 1) Leer nube
        if ext == '.bin':
            points, remissions = read_bin_file(input_path)
        elif ext == '.ply':
            points, remissions = read_ply_file(input_path)
        else:
            points, remissions = read_pcd_file(input_path)

        # 2) Leer etiquetas originales
        if label_dir:
            parts = base.split('_')
            common = "_".join(parts[:-1]) if len(parts) > 1 else base
            cand = glob.glob(os.path.join(label_dir, f"*{common}*.label"))
            if cand:
                labels = np.fromfile(cand[0], dtype=np.uint32)
                if len(labels) != len(points):
                    print(f"⚠ Mismatch: {file} tiene {len(points)} pts vs {len(labels)} lbls")
                    labels = np.zeros(len(points), dtype=np.uint32)
            else:
                print(f"⚠ No se encontró .label para {file}")
                labels = np.zeros(len(points), dtype=np.uint32)
        else:
            labels = np.zeros(len(points), dtype=np.uint32)

        print(f"\n🔍 Procesando: {file} ({len(points)} puntos)")

        # 3) Densificar con etiquetas (pasa labels)
        start_time = time.time()
        dense_pts, dense_rem, dense_lbl = densify_point_cloud(
            points, remissions, labels, density_factor
        )
        elapsed = time.time() - start_time
        print(f"✅ Densificado a {len(dense_pts)} puntos (took {elapsed:.2f}s)")

        # 4) Guardar nube densificada
        out_pc = os.path.join(output_dir, f"densified_{file}")
        save_point_cloud(dense_pts, dense_rem, out_pc, ext)
        print(f"💾 Guardado como: densified_{file}")

        # 5) Guardar etiquetas densificadas
        out_lbl = os.path.splitext(out_pc)[0] + ".label"
        dense_lbl.astype(np.uint32).tofile(out_lbl)
        print(f"💾 Etiquetas guardadas en: {os.path.basename(out_lbl)}")

    print("\n✅ Proceso completado!")

def main():

    print("\n" + "="*50)
    print(" DENSIFICADOR DE NUBES DE PUNTOS LIDAR (3 PUNTOS)")
    print("="*50 + "\n")

    # Solicitar directorio de entrada
    while True:
        input_dir = input("📂 Introduzca la ruta del directorio de entrada: ").strip()
        if os.path.isdir(input_dir):
            break
        print("⚠ El directorio no existe. Intente nuevamente.")

    # Solicitar directorio de etiquetas
    while True:
        label_dir = input("🏷️ Introduzca la ruta del directorio de etiquetas (.label) (opcional): ").strip()
        if label_dir == "":
            label_dir = None
            break
        elif os.path.isdir(label_dir):
            break
        print("⚠ El directorio no existe. Intente nuevamente.")

    # Solicitar factor de densificación
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
    print(f" - Carpeta de salida: {os.path.join(input_dir, f'densified_IDW_3points_{density_factor}')}")

    # Pasar label_dir al batch
    batch_densification(input_dir, density_factor, label_dir)


if __name__ == "__main__":
    main()