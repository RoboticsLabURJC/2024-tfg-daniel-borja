import numpy as np
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
import sys
from typing import Tuple
from scipy.spatial import Delaunay
from tqdm import tqdm 
import time
from scipy.spatial import KDTree
import glob

# ------ Lectura de archivos -------
def read_pcd_file(file_path):
    """Lee archivos .pcd y devuelve los puntos e intensidades."""
    if not file_path.endswith('.pcd') or not os.path.exists(file_path):
        print(f"Error: .pcd file not found at {file_path}")
        sys.exit(1)
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    
    if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
        remissions = np.linalg.norm(np.asarray(pcd.colors), axis=1)
    else:
        remissions = np.zeros(len(points))
    
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

# ------ Adición de Ruido Gaussiano -------
def add_gaussian_noise(points: np.ndarray, remissions: np.ndarray, labels: np.ndarray,
                       std_dev: float, points_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Añade ruido gaussiano a una fracción de puntos y mantiene sus etiquetas originales.
    """
    print("\n=== INICIO DEL PROCESO DE AÑADIR RUIDO GAUSSIANO ===")
    
    num_points = len(points)
    num_noisy_points = int(num_points * points_factor)

    print(f"Puntos totales: {num_points}, Puntos a modificar: {num_noisy_points}")
    
    rng = np.random.default_rng()
    selected_indices = rng.choice(num_points, size=num_noisy_points, replace=False)

    # Copiamos los arrays originales
    noisy_points = points.copy()
    noisy_remissions = remissions.copy()
    noisy_labels = labels.copy()  # Esto ya contiene las etiquetas originales

    # Generamos ruido y lo aplicamos solo a puntos seleccionados
    noise = rng.normal(0, std_dev, (num_noisy_points, 3))
    noisy_points[selected_indices] += noise

    #intensity_noise = rng.normal(1, std_dev * 0.5, num_noisy_points)
    #noisy_remissions[selected_indices] *= np.clip(intensity_noise, 0.5, 1.5)

    # 🔒 Las etiquetas se mantienen sin cambios
    print("Etiquetas mantenidas sin cambios.")

    return noisy_points, noisy_remissions, noisy_labels


# ------ Guardado de archivos -------
def save_point_cloud(points: np.ndarray, remissions: np.ndarray, 
                    output_path: str, input_extension: str):
    """Guarda la nube de puntos en el formato original."""
    if input_extension == '.ply':
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
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if remissions is not None and len(remissions) > 0:
            colors = np.tile(remissions[:, None], (1, 3)) / np.max(remissions)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
        print(f"Archivo PCD guardado en: {output_path}")
    
    elif input_extension == '.bin':
        combined = np.hstack((points, remissions[:, None]))
        combined.astype(np.float32).tofile(output_path)
        print(f"Archivo BIN guardado en: {output_path}")
    
    else:
        print(f"Formato '{input_extension}' no soportado.")

# ------ Procesamiento por lotes -------
def batch_add_noise_with_labels(input_dir: str, std_dev: float, points_factor: float,
                                label_dir: str = None):
    output_dir = os.path.join(input_dir, f"noisy_{std_dev}")
    os.makedirs(output_dir, exist_ok=True)

    input_files = [f for f in os.listdir(input_dir) if f.endswith(('.bin', '.ply', '.pcd'))]
    if not input_files:
        print("No se encontraron archivos válidos.")
        return

    for file in tqdm(input_files, desc="Procesando archivos"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"noisy_{file}")
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
                # Extraer parte común del nombre antes del último guion bajo
                parts = base_name.split('_')
                if len(parts) > 1:
                    base_name_common = "_".join(parts[:-1])
                else:
                    base_name_common = base_name

                # Buscar archivo .label en label_dir que contenga esa parte común
                possible_labels = glob.glob(os.path.join(label_dir, f"*{base_name_common}*.label"))

                if len(possible_labels) == 0:
                    print(f"⚠ No se encontró archivo .label para {file} (buscando con patrón '{base_name_common}')")
                    labels = np.zeros(len(points), dtype=np.uint32)  # Poner etiquetas vacías para no romper
                else:
                    label_path = possible_labels[0]
                    labels = np.fromfile(label_path, dtype=np.uint32)
                    if len(labels) != len(points):
                        print(f"⚠ Mismatch: {file} tiene {len(points)} puntos, pero {len(labels)} etiquetas.")
                        labels = np.zeros(len(points), dtype=np.uint32)

            else:
                labels = np.zeros(len(points), dtype=np.uint32)

            # Añadir ruido sin modificar etiquetas
            noisy_points, noisy_remissions, noisy_labels = add_gaussian_noise(
                points, remissions, labels, std_dev, points_factor)

            save_point_cloud(noisy_points, noisy_remissions, output_path, ext)

            # Guardar etiquetas originales sin cambio
            if label_dir and labels is not None:
                label_output_path = os.path.splitext(output_path)[0] + ".label"
                noisy_labels.astype(np.uint32).tofile(label_output_path)
                print(f"Archivo de etiquetas guardado en: {label_output_path}")

        except Exception as e:
            print(f"❌ Error procesando {file}: {str(e)}")

    print("✅ Proceso de añadir ruido completado.")


# ------ Interfaz de usuario -------
def main():
    print("\n=== AÑADIR RUIDO GAUSSIANO CON ETIQUETAS ===")

    points_factor = 1

    while True:
        input_dir = input("📁 Directorio con archivos de entrada (.bin, .pcd, .ply): ").strip()
        if os.path.isdir(input_dir):
            break
        print("⚠ Directorio no válido.")

    while True:
        label_dir = input("🏷️ Directorio con archivos .label (opcional): ").strip()
        if label_dir == "":
            label_dir = None
            break
        elif os.path.isdir(label_dir):
            break
        print("⚠ Directorio no válido.")

    while True:
        try:
            std_dev = float(input("σ Introduzca desviación estándar del ruido (ej. 0.05): "))
            if std_dev > 0:
                break
            print("⚠ Debe ser mayor que 0.")
        except ValueError:
            print("⚠ Debe introducir un número válido.")
        
    
    batch_add_noise_with_labels(input_dir, std_dev, points_factor, label_dir)


if __name__ == "__main__":
    main()