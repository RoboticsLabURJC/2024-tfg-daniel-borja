import numpy as np
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
import sys
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import filedialog, simpledialog
from scipy.stats import truncnorm
from scipy.spatial import KDTree
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

def densify_point_cloud(points: np.ndarray, remissions: np.ndarray, 
                       density_factor: float = 5.0, search_radius: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Versión optimizada para 3 puntos con KDTree.
    """
    tree = KDTree(points)  # Estructura eficiente para búsquedas
    
    num_new_points = int((len(points) * (density_factor-1)))
    point_usage_count = {}
    new_points = []
    new_remissions = []

    # Parámetros distribución
    media, desviacion = 0.5, 0.2
    a, b = (0 - media)/desviacion, (1 - media)/desviacion
    
    for _ in range(num_new_points):
        # Selección de punto igual que antes
        if not point_usage_count:
            index = np.random.randint(0, len(points))
        else:
            min_usage = min(point_usage_count.values())
            candidates = [i for i, cnt in point_usage_count.items() if cnt == min_usage]
            candidates += [i for i in range(len(points)) if i not in point_usage_count]
            index = np.random.choice(candidates)

        point = points[index]
        point_usage_count[index] = point_usage_count.get(index, 0) + 1

        # Búsqueda por radio optimizada
        neighbor_indices = tree.query_ball_point(point, search_radius)
        
        if len(neighbor_indices) < 3:  # Necesitamos 3 puntos
            continue
            
        # Obtener los 3 más cercanos dentro del radio
        distances, indices = tree.query(point, k=3)
        
        # Puntos para interpolación
        A, B, C = points[indices[0]], points[indices[1]], points[indices[2]]

        # Pesos aleatorios normalizados
        lambdas = truncnorm.rvs(a, b, loc=media, scale=desviacion, size=3)
        lambdas /= lambdas.sum()

        # Interpolación
        new_point = lambdas[0]*A + lambdas[1]*B + lambdas[2]*C

        # Cálculo de distancias e intensidades
        d_A, d_B, d_C = [np.linalg.norm(new_point - p) for p in [A, B, C]]
        r_A, r_B, r_C = remissions[indices[0]], remissions[indices[1]], remissions[indices[2]]
        
        new_remission = ((1/d_A)*r_A + (1/d_B)*r_B + (1/d_C)*r_C) / ((1/d_A) + (1/d_B) + (1/d_C))
        
        new_points.append(new_point)
        new_remissions.append(new_remission)

    # Combinar resultados
    combined_points = np.vstack([points, np.array(new_points)])
    combined_remissions = np.concatenate([remissions, np.array(new_remissions)])
    
    return combined_points, combined_remissions

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


def batch_densification_with_viz(input_directory: str, output_directory: str, 
                                  density_factor: float = 5.0):
    os.makedirs(output_directory, exist_ok=True)
    
    input_files = [f for f in os.listdir(input_directory) 
                   if f.endswith(('.bin', '.ply', '.pcd'))]
    
    for file in input_files:
        input_path = os.path.join(input_directory, file)
        output_filename = f"densified_{file}"  # Mantener misma extensión
        output_path = os.path.join(output_directory, output_filename)
        input_extension = os.path.splitext(file)[1]  # Obtener extensión (.bin, .ply, .pcd)
        
        try:
            # Leer archivo dependiendo de su extensión
            if input_extension == '.bin':
                points, remissions = read_bin_file(input_path)
            elif input_extension == '.ply':
                points, remissions = read_ply_file(input_path)
            elif input_extension == '.pcd':
                points, remissions = read_pcd_file(input_path)
            else:
                print(f"Formato '{input_extension}' no soportado para lectura.")
                continue
            
            print(f"Número de puntos original: {len(points)}")
            
            # Densificar la nube de puntos
            points, remissions = densify_point_cloud(
                points, remissions, density_factor
            )
            
            print(f"Número de puntos densificado: {len(points)}")
            
            # Guardar los puntos densificados en el mismo formato
            save_point_cloud(
                np.vstack([points]), 
                np.concatenate([remissions]), 
                output_path,
                input_extension
            )
        
        except Exception as e:
            print(f"Error procesando {file}: {e}")

            

def main():
    # Solicitar directorio de entrada por terminal
    print("\n=== Densificación de nubes de puntos con IDW(3-NN) ===")
    input_directory = input("Introduce la ruta del directorio de entrada: ").strip()
    if not os.path.isdir(input_directory):
        print(f"\nError: El directorio '{input_directory}' no existe.")
        return

    # Solicitar factor de densificación con validación
    while True:
        try:
            density_factor = float(input("Introduce el factor de densificación (recomendado 3-10): "))
            if 1.0 <= density_factor <= 20.0:
                break
            else:
                print("Error: El valor debe estar entre 1.0 y 20.0")
        except ValueError:
            print("Error: Introduce un número válido.")

    # Crear directorio de salida automáticamente dentro del directorio de entrada
    output_dir_name = f"densified_IDW_3points_{density_factor}"
    output_directory = os.path.join(input_directory, output_dir_name)
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"\nSe creó el directorio de salida: {output_directory}")
    print("\nIniciando procesamiento...")
   
    # Ejecutar el proceso de densificación
    batch_densification_with_viz(
        input_directory,
        output_directory,
        density_factor
    )
    
    print("\n=== Proceso completado ===")
    print(f"Resultados guardados en: {output_directory}")

if __name__ == "__main__":
    main()
