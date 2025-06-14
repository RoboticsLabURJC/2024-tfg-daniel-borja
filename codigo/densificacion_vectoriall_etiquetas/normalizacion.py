import numpy as np
import os
import glob
import sys
import open3d as o3d
from plyfile import PlyData, PlyElement
from typing import Tuple
from tqdm import tqdm

# ------ Lectura de archivos -------

def read_pcd_file(file_path):
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
    if not file_path.endswith('.bin') or not os.path.exists(file_path):
        print(f"Error: .bin file not found at {file_path}")
        sys.exit(1)
    scan = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
    points, remissions = scan[:, :3], scan[:, 3]
    return points, remissions

def read_ply_file(file_path):
    if not file_path.endswith('.ply') or not os.path.exists(file_path):
        print(f"Error: .ply file not found at {file_path}")
        sys.exit(1)
    plydata = PlyData.read(file_path)
    x, y, z = plydata['vertex'].data['x'], plydata['vertex'].data['y'], plydata['vertex'].data['z']
    points = np.vstack((x, y, z)).T
    remissions = plydata['vertex'].data['intensity']
    return points, remissions

# ------ Filtrado y submuestreo -------

def filter_and_subsample(points: np.ndarray, remissions: np.ndarray,
                         max_distance: float = 25.0, num_points: int = None,
                         seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filtra puntos a max_distance y submuestrea num_points puntos.
    
    :param points: (N,3) puntos originales
    :param remissions: (N,) intensidades
    :param max_distance: distancia máxima para filtrar
    :param num_points: número de puntos finales a seleccionar dentro del rango
    :param seed: semilla para reproducibilidad
    :return: puntos filtrados y submuestreados, remissions, indices originales
    """
    distances = np.linalg.norm(points, axis=1)
    mask = distances <= max_distance
    filtered_points = points[mask]
    filtered_remissions = remissions[mask]
    filtered_indices = np.where(mask)[0]

    if num_points is not None and filtered_points.shape[0] >= num_points:
        rng = np.random.default_rng(seed)
        chosen_idx = rng.choice(filtered_points.shape[0], size=num_points, replace=False)
        final_points = filtered_points[chosen_idx]
        final_remissions = filtered_remissions[chosen_idx]
        final_indices = filtered_indices[chosen_idx]
    else:
        # Si no se especifica num_points o hay menos puntos que num_points, devuelve todos
        final_points = filtered_points
        final_remissions = filtered_remissions
        final_indices = filtered_indices

    return final_points, final_remissions, final_indices

# ------ Guardado de archivos -------

def save_point_cloud(points: np.ndarray, remissions: np.ndarray, output_path: str, input_extension: str):
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
            colors = np.tile(remissions[:, None], (1, 3))
            colors = colors / np.max(colors)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
        print(f"Archivo PCD guardado en: {output_path}")

    elif input_extension == '.bin':
        combined = np.hstack((points, remissions[:, None]))
        combined.astype(np.float32).tofile(output_path)
        print(f"Archivo BIN guardado en: {output_path}")

    else:
        print(f"Formato '{input_extension}' no soportado. No se guardó el archivo.")

# ------ Procesamiento por lotes -------

def batch_process(input_dir: str, max_distance: float = 25.0, num_points: int = None,
                  label_dir: str = None):
    output_dir = os.path.join(input_dir, f"filtered_within_{max_distance}m")
    os.makedirs(output_dir, exist_ok=True)

    input_files = [f for f in os.listdir(input_dir) if f.endswith(('.bin', '.ply', '.pcd'))]
    if not input_files:
        print("No se encontraron archivos válidos.")
        return

    for file in tqdm(input_files, desc="Procesando archivos"):
        try:
            input_path = os.path.join(input_dir, file)
            ext = os.path.splitext(file)[1]
            base_name = os.path.splitext(file)[0]
            output_path = os.path.join(output_dir, f"filtered_{file}")

            # Leer nube y remisiones
            if ext == '.bin':
                points, remissions = read_bin_file(input_path)
            elif ext == '.ply':
                points, remissions = read_ply_file(input_path)
            elif ext == '.pcd':
                points, remissions = read_pcd_file(input_path)
            else:
                print(f"Formato {ext} no soportado.")
                continue

            # Filtrar y submuestrear
            filtered_points, filtered_remissions, indices = filter_and_subsample(
                points, remissions, max_distance=max_distance, num_points=num_points, seed=42
            )

            save_point_cloud(filtered_points, filtered_remissions, output_path, ext)

            # Procesar etiquetas
            if label_dir:
                base_name_original = base_name
                if base_name.startswith('filtered_'):
                    base_name_original = base_name[len('filtered_'):]
                parts = base_name_original.split('_')
                if len(parts) > 1:
                    base_name_common = "_".join(parts[:-1])
                else:
                    base_name_common = base_name_original

                possible_labels = glob.glob(os.path.join(label_dir, f"*{base_name_common}*.label"))
                if len(possible_labels) == 0:
                    print(f"⚠ No se encontró archivo .label para {file} (buscando con patrón '{base_name_common}')")
                else:
                    label_path = possible_labels[0]
                    labels = np.fromfile(label_path, dtype=np.uint32)
                    if len(labels) != len(points):
                        print(f"⚠ Mismatch: {file} tiene {len(points)} puntos, pero {len(labels)} etiquetas.")
                        continue
                    subsampled_labels = labels[indices]
                    label_output_path = os.path.splitext(output_path)[0] + ".label"
                    subsampled_labels.astype(np.uint32).tofile(label_output_path)

        except Exception as e:
            print(f"❌ Error procesando {file}: {e}")

    print("✅ Procesamiento completado.")

# ------ Interfaz de usuario -------

def main():
    print("\n=== FILTRADO Y SUBMUESTREO CON ETIQUETAS ===")

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
            max_distance = float(input("📏 Distancia máxima para filtrar puntos (ej. 25.0): "))
            if max_distance > 0:
                break
            print("⚠ Debe ser un número positivo.")
        except ValueError:
            print("⚠ Ingresa un número válido.")

    while True:
        num_points_input = input("🔢 Número de puntos a conservar dentro del rango (deja vacío para todos): ").strip()
        if num_points_input == "":
            num_points = None
            break
        try:
            num_points = int(num_points_input)
            if num_points > 0:
                break
            print("⚠ Debe ser un entero positivo.")
        except ValueError:
            print("⚠ Ingresa un número válido.")

    batch_process(input_dir, max_distance=max_distance, num_points=num_points, label_dir=label_dir)

if __name__ == "__main__":
    main()
