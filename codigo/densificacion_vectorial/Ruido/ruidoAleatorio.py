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
def add_gaussian_noise(points: np.ndarray, remissions: np.ndarray, 
                      std_dev: float, points_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Añade ruido gaussiano a la nube de puntos.
    
    :param points: Puntos originales (N, 3)
    :param remissions: Intensidades originales (N,)
    :param std_dev: Desviación estándar del ruido (metros)
    :param points_factor: Porcentaje de puntos a modificar (0-1)
    :return: Puntos con ruido e intensidades modificadas
    """
    print("\n=== INICIO DEL PROCESO DE AÑADIR RUIDO GAUSSIANO ===")
    print(f"Puntos iniciales: {points.shape}, Remisiones: {remissions.shape}")
    
    # Validación de parámetros
    if std_dev <= 0:
        raise ValueError("La desviación estándar debe ser positiva")
    if points_factor <= 0 or points_factor > 1:
        raise ValueError("El factor de puntos debe estar entre 0 y 1")
    
    num_points = len(points)
    num_noisy_points = int(num_points * points_factor)
    
    print(f"\n1. Seleccionando {num_noisy_points} puntos para añadir ruido...")
    rng = np.random.default_rng()
    selected_indices = rng.choice(num_points, size=num_noisy_points, replace=False)
    
    print(f"\n2. Generando ruido gaussiano con σ={std_dev}...")
    noise = rng.normal(0, std_dev, (num_noisy_points, 3))
    
    print("\n3. Aplicando ruido a los puntos seleccionados...")
    noisy_points = points.copy()
    noisy_points[selected_indices] += noise
    
    print("\n4. Ajustando intensidades con ruido proporcional...")
    noisy_remissions = remissions.copy()
    intensity_noise = rng.normal(1, std_dev*0.5, num_noisy_points)
    noisy_remissions[selected_indices] *= np.clip(intensity_noise, 0.5, 1.5)
    
    print("\n=== RESULTADOS ===")
    print(f"Total puntos originales: {num_points}")
    print(f"Puntos modificados: {num_noisy_points}")
    print(f"Desviación estándar aplicada: {std_dev} metros")
    
    return noisy_points, noisy_remissions

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
def batch_add_noise(input_dir: str, output_dir: str, std_dev: float, points_factor: float):
    """Procesa todos los archivos en el directorio de entrada."""
    os.makedirs(output_dir, exist_ok=True)
    
    input_files = [f for f in os.listdir(input_dir) 
                  if f.endswith(('.bin', '.ply', '.pcd'))]
    
    if not input_files:
        print("\n⚠ No se encontraron archivos .bin, .ply o .pcd")
        return
    
    print(f"\n📁 Archivos a procesar ({len(input_files)}):")
    for i, f in enumerate(input_files, 1):
        print(f" {i}. {f}")
    
    for file in tqdm(input_files, desc="Procesando archivos"):
        try:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, f"noisy_{file}")
            ext = os.path.splitext(file)[1]
            
            print(f"\n🔍 Procesando: {file}")
            if ext == '.bin':
                points, remissions = read_bin_file(input_path)
            elif ext == '.ply':
                points, remissions = read_ply_file(input_path)
            elif ext == '.pcd':
                points, remissions = read_pcd_file(input_path)
            
            print(f"Puntos originales: {len(points):,}")
            
            start_time = time.time()
            noisy_points, noisy_remissions = add_gaussian_noise(
                points, remissions, std_dev, points_factor
            )
            elapsed = time.time() - start_time
            
            print(f"✅ Ruido añadido en {elapsed:.2f}s")
            save_point_cloud(noisy_points, noisy_remissions, output_path, ext)
            
        except Exception as e:
            print(f"\n❌ Error procesando {file}: {str(e)}")
    
    print("\n✅ Proceso completado!")

# ------ Interfaz de usuario -------
def main():
    print("\n" + "="*50)
    print(" HERRAMIENTA DE AÑADIR RUIDO GAUSSIANO A NUBES DE PUNTOS")
    print("="*50 + "\n")

    # Solicitar directorio de entrada
    while True:
        input_dir = input("📂 Introduzca la ruta del directorio de entrada: ").strip()
        if os.path.isdir(input_dir):
            break
        print("⚠ El directorio no existe. Intente nuevamente.")

    # Crear directorio de salida
    output_dir = os.path.join(input_dir, "noisy_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📂 Los resultados se guardarán en: {output_dir}")

    # Solicitar desviación estándar
    while True:
        try:
            std_dev = float(input("\nσ Introduzca la desviación estándar del ruido (en metros, ej. 0.05): "))
            if std_dev > 0:
                break
            print("⚠ El valor debe ser mayor que 0")
        except ValueError:
            print("⚠ Debe introducir un número válido")

    # Solicitar factor de puntos
    while True:
        try:
            points_factor = float(input("\n🔢 Introduzca el porcentaje de puntos a modificar (0-1, ej. 0.8): "))
            if 0 < points_factor <= 1:
                break
            print("⚠ El valor debe estar entre 0 y 1")
        except ValueError:
            print("⚠ Debe introducir un número válido")

    # Confirmación
    print(f"\n⚙ Configuración final:")
    print(f" - Directorio de entrada: {input_dir}")
    print(f" - Directorio de salida: {output_dir}")
    print(f" - Desviación estándar: {std_dev} metros")
    print(f" - Porcentaje de puntos a modificar: {points_factor*100:.1f}%")
    
    confirm = input("\n¿Desea continuar? (s/n): ").lower()
    if confirm == 's':
        batch_add_noise(input_dir, output_dir, std_dev, points_factor)
    else:
        print("\nProceso cancelado.")

if __name__ == "__main__":
    main()