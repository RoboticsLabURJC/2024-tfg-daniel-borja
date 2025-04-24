import numpy as np
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
import sys
from typing import Tuple
from tqdm import tqdm 
import time

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
    
    return subsampled_points, subsampled_remissions

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

def batch_subsampling(input_dir: str, reduction_factor: float = 0.5):
    """Procesa todos los archivos en un directorio aplicando submuestreo aleatorio."""
    # Crear nombre de carpeta basado en el factor de reducción
    subsampled_folder_name = f"subsampled_{reduction_factor}"
    output_dir = os.path.join(input_dir, subsampled_folder_name)

    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n Los archivos submuestreados se guardarán en: {output_dir}")

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
        output_path = os.path.join(output_dir, f"subsampled_{file}")
        ext = os.path.splitext(file)[1]
        
        try:
            # Leer archivo
            if ext == '.bin':
                points, remissions = read_bin_file(input_path)
            elif ext == '.ply':
                points, remissions = read_ply_file(input_path)
            elif ext == '.pcd':
                points, remissions = read_pcd_file(input_path)
            
            print(f"\n🔍 Procesando: {file} ({len(points)} puntos)")
            
            # Aplicar submuestreo
            start_time = time.time()
            subsampled_points, subsampled_remissions = subsample_point_cloud(points, remissions, reduction_factor)
            elapsed = time.time() - start_time
            
            print(f"✅ Submuestreado a {len(subsampled_points)} puntos (took {elapsed:.2f}s)")
            
            # Guardar
            save_point_cloud(subsampled_points, subsampled_remissions, output_path, ext)
            print(f"💾 Guardado como: subsampled_{file}")
            
        except Exception as e:
            print(f"\n❌ Error procesando {file}: {str(e)}")
    
    print("\n Proceso completado!")

# ------ Interfaz de usuario -------

def main():
    print("\n" + "="*50)
    print(" SUBMUESTREADOR ALEATORIO DE NUBES DE PUNTOS LIDAR")
    print("="*50 + "\n")

    # Solicitar directorio de entrada
    while True:
        input_dir = input("📂 Introduzca la ruta del directorio de entrada: ").strip()
        if os.path.isdir(input_dir):
            break
        print("⚠ El directorio no existe. Intente nuevamente.")

    # Solicitar factor de reducción
    while True:
        try:
            reduction_factor = float(input("\n🔢 Introduzca el factor de reducción (ej. 0.5 para reducir a la mitad): "))
            if 0 < reduction_factor < 1:
                break
            print("⚠ El valor debe estar entre 0 y 1")
        except ValueError:
            print("⚠ Debe introducir un número válido")

    # Confirmación
    print(f"\n⚙ Configuración:")
    print(f" - Directorio de entrada: {input_dir}")
    print(f" - Factor de reducción: {reduction_factor}")
    print(f" - Los resultados se guardarán en: {os.path.join(input_dir, f'subsampled_{reduction_factor}')}")
    
    batch_subsampling(input_dir, reduction_factor)

if __name__ == "__main__":
    main()