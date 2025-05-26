import numpy as np
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
import sys
from typing import Tuple
from tqdm import tqdm 
import time
import glob


# ------ Lectura de archivos ------- (igual que en TIN.py)

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

# ------ Submuestreo aleatorio -------

def subsample_point_cloud(points: np.ndarray, remissions: np.ndarray, 
                         reduction_factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce la nube de puntos mediante submuestreo aleatorio.
    
    :param points: Nube de puntos original (N, 3)
    :param remissions: Intensidades de los puntos originales (N,)
    :param reduction_factor: Factor de reducción (0 < reduction_factor < 1)
    :return: Puntos reducidos y sus intensidades
    """
    print("\n=== INICIO DEL PROCESO DE SUBMUESTREO ALEATORIO ===")
    print(f"Puntos iniciales: {points.shape}, Remisiones: {remissions.shape}")

    # Validar factor de reducción
    if reduction_factor <= 0 or reduction_factor >= 1:
        raise ValueError("El factor de reducción debe estar entre 0 y 1")
    
    # Calcular número de puntos a conservar
    num_original_points = len(points)
    num_subsampled_points = int(num_original_points * reduction_factor)
    print(f"\n1. Reduciendo de {num_original_points} a {num_subsampled_points} puntos...")

    # Generar índices aleatorios sin repetición
    print("\n2. Generando índices aleatorios...")
    rng = np.random.default_rng()
    indices = rng.choice(num_original_points, size=num_subsampled_points, replace=False)
    indices = np.sort(indices)  # Ordenar para mantener cierta coherencia espacial
    
    print("\nÍndices seleccionados (primeros 10):")
    print(indices[:10])

    # Seleccionar puntos e intensidades
    print("\n3. Seleccionando puntos e intensidades...")
    subsampled_points = points[indices]
    subsampled_remissions = remissions[indices]

    print("\nPuntos submuestreados (primeros 10):")
    print(subsampled_points[:10])
    print("\nRemisiones submuestreadas (primeras 10):")
    print(subsampled_remissions[:10])

    print("\n=== RESULTADOS FINALES ===")
    print(f"Total puntos originales: {len(points)}")
    print(f"Total puntos después del submuestreo: {len(subsampled_points)}")
    print("\nPrimeros 5 puntos originales:")
    print(points[:5])
    print("\nPrimeros 5 puntos después del submuestreo:")
    print(subsampled_points[:5])
    
    return subsampled_points, subsampled_remissions, indices

# ------ Guardado de archivos ------- (igual que en TIN.py)

def save_point_cloud(points: np.ndarray, remissions: np.ndarray, output_path: str, input_extension: str):
    """
    Guarda la nube de puntos submuestreada en el formato especificado.
    """
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
        print(f"Formato '{input_extension}' no soportado. No se guardó el archivo.")

# ------ Procesamiento por lotes -------

def batch_subsampling(input_dir: str, reduction_factor: float = 0.5, label_dir: str = None):
    output_dir = os.path.join(input_dir, f"subsampled_{reduction_factor}")
    os.makedirs(output_dir, exist_ok=True)

    input_files = [f for f in os.listdir(input_dir) if f.endswith(('.bin', '.ply', '.pcd'))]
    if not input_files:
        print("No se encontraron archivos válidos.")
        return

    for file in tqdm(input_files, desc="Procesando archivos"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"subsampled_{file}")
        ext = os.path.splitext(file)[1]
        base_name = os.path.splitext(file)[0]

        try:
            # Leer puntos
            if ext == '.bin':
                points, remissions = read_bin_file(input_path)
            elif ext == '.ply':
                points, remissions = read_ply_file(input_path)
            elif ext == '.pcd':
                points, remissions = read_pcd_file(input_path)

            # Submuestreo
            subsampled_points, subsampled_remissions, indices = subsample_point_cloud(points, remissions, reduction_factor)
            save_point_cloud(subsampled_points, subsampled_remissions, output_path, ext)

            # Leer y guardar etiquetas si están disponibles
            if label_dir:
                # Ajustar base_name para eliminar prefijo 'subsampled_' si existe
                base_name_original = base_name
                if base_name.startswith('subsampled_'):
                    base_name_original = base_name[len('subsampled_'):]
                
                # Extraer parte común del nombre antes del último guion bajo
                # ej: '2022-07-22_flight__0071_1658494234334310308_vls128' -> '2022-07-22_flight__0071_1658494234334310308'
                parts = base_name_original.split('_')
                if len(parts) > 1:
                    base_name_common = "_".join(parts[:-1])
                else:
                    base_name_common = base_name_original

                # Buscar archivo .label en label_dir que contenga esa parte común
                possible_labels = glob.glob(os.path.join(label_dir, f"*{base_name_common}*.label"))
                if len(possible_labels) == 0:
                    print(f"⚠ No se encontró archivo .label para {file} (buscando con patrón '{base_name_common}')")
                else:
                    label_path = possible_labels[0]  # Tomar el primero que coincida
                    labels = np.fromfile(label_path, dtype=np.uint32)
                    if len(labels) != len(points):
                        print(f"⚠ Mismatch: {file} tiene {len(points)} puntos, pero {len(labels)} etiquetas.")
                        continue
                    subsampled_labels = labels[indices]
                    label_output_path = os.path.splitext(output_path)[0] + ".label"
                    subsampled_labels.astype(np.uint32).tofile(label_output_path)

        except Exception as e:
            print(f"❌ Error procesando {file}: {str(e)}")

    print("✅ Submuestreo completado.")

# ------ Interfaz de usuario -------

def main():
    print("\n=== SUBMUESTREADOR CON ETIQUETAS ===")

    while True:
        input_dir = input("📁 Directorio con archivos de entrada (.bin, .pcd, .ply): ").strip()
        if os.path.isdir(input_dir):
            break
        print("⚠ Directorio no válido.")

    while True:
        label_dir = input("🏷️  Directorio con archivos .label (opcional): ").strip()
        if label_dir == "":
            label_dir = None
            break
        elif os.path.isdir(label_dir):
            break
        print("⚠ Directorio no válido.")

    while True:
        try:
            reduction_factor = float(input("🔽 Factor de reducción (ej. 0.5): "))
            if 0 < reduction_factor < 1:
                break
            print("⚠ Debe estar entre 0 y 1.")
        except ValueError:
            print("⚠ Ingresa un número válido.")

    batch_subsampling(input_dir, reduction_factor, label_dir)

if __name__ == "__main__":
    main()