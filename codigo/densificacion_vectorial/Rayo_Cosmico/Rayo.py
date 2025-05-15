import numpy as np
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
import sys
from typing import Tuple
from tqdm import tqdm 
import time

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
    points, remissions = scan[:, 0:3], scan[:, 3]
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

def apply_cosmic_ray(points: np.ndarray, remissions: np.ndarray, 
                      impact_probability: float) -> Tuple[np.ndarray, np.ndarray]:
    print("\n=== IMPACTO DE RAYO CÓSMICO EN PROGRESO ===")
    if impact_probability <= 0 or impact_probability > 1:
        raise ValueError("La probabilidad de impacto debe estar entre 0 y 1")

    num_points = len(points)
    num_impacted = int(num_points * impact_probability)

    print(f"Afectando {num_impacted} de {num_points} puntos...")
    rng = np.random.default_rng()
    impacted_indices = rng.choice(num_points, size=num_impacted, replace=False)

    displaced_points = points.copy()
    displaced_remissions = remissions.copy()

    displacement = rng.uniform(-2.0, 2.0, size=(num_impacted, 3))
    intensity_scale = rng.uniform(0.1, 3.0, size=num_impacted)

    displaced_points[impacted_indices] += displacement
    displaced_remissions[impacted_indices] *= intensity_scale
    displaced_remissions = np.clip(displaced_remissions, 0.0, 255.0)

    print("Impacto completado con desplazamientos y cambios de remisión.")
    return displaced_points, displaced_remissions

def save_point_cloud(points: np.ndarray, remissions: np.ndarray, 
                     output_path: str, input_extension: str):
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

def batch_cosmic_strike(input_dir: str, output_dir: str, impact_probability: float):
    os.makedirs(output_dir, exist_ok=True)
    input_files = [f for f in os.listdir(input_dir) 
                   if f.endswith(('.bin', '.ply', '.pcd'))]
    if not input_files:
        print("\n⚠ No se encontraron archivos .bin, .ply o .pcd")
        return

    print(f"\n📁 Archivos a procesar ({len(input_files)}):")
    for i, f in enumerate(input_files, 1):
        print(f" {i}. {f}")

    for file in tqdm(input_files, desc="Aplicando rayos cósmicos"):
        try:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, f"cosmic_{file}")
            ext = os.path.splitext(file)[1]

            if ext == '.bin':
                points, remissions = read_bin_file(input_path)
            elif ext == '.ply':
                points, remissions = read_ply_file(input_path)
            elif ext == '.pcd':
                points, remissions = read_pcd_file(input_path)

            print(f"\n🔍 Procesando: {file}")
            start_time = time.time()
            new_points, new_remissions = apply_cosmic_ray(points, remissions, impact_probability)
            elapsed = time.time() - start_time

            print(f"✅ Rayos aplicados en {elapsed:.2f}s")
            save_point_cloud(new_points, new_remissions, output_path, ext)

        except Exception as e:
            print(f"\n❌ Error procesando {file}: {str(e)}")

    print("\n✅ Proceso completado!")

def main():
    print("\n" + "="*50)
    print(" SIMULADOR DE IMPACTO DE RAYO CÓSMICO EN NUBES LIDAR")
    print("="*50 + "\n")

    while True:
        input_dir = input("📂 Introduzca la ruta del directorio de entrada: ").strip()
        if os.path.isdir(input_dir):
            break
        print("⚠ El directorio no existe. Intente nuevamente.")

    output_dir = os.path.join(input_dir, "cosmic_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📂 Los resultados se guardarán en: {output_dir}")

    while True:
        try:
            impact_prob = float(input("\n🌠 Introduzca la probabilidad de impacto (0-1, ej. 0.3): "))
            if 0 < impact_prob <= 1:
                break
            print("⚠ El valor debe estar entre 0 y 1")
        except ValueError:
            print("⚠ Debe introducir un número válido")

    print(f"\n⚙ Configuración:")
    print(f" - Probabilidad de impacto: {impact_prob*100:.1f}%")
    print(f" - Directorio de entrada: {input_dir}")
    print(f" - Directorio de salida: {output_dir}")

    confirm = input("\n¿Desea continuar? (s/n): ").lower()
    if confirm == 's':
        batch_cosmic_strike(input_dir, output_dir, impact_prob)
    else:
        print("\nProceso cancelado.")

if __name__ == "__main__":
    main()
