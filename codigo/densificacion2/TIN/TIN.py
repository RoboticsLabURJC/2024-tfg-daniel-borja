import numpy as np
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
import sys
from typing import Tuple
from scipy.spatial import Delaunay
import tkinter as tk
from tkinter import filedialog, simpledialog

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

# ------ Densificación con Triangulación de Delaunay -------

def densify_point_cloud(points: np.ndarray, remissions: np.ndarray, 
                                       density_factor: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Densifica la nube de puntos utilizando triangulación de Delaunay e interpolación lineal.
    """
    # Crear la triangulación de Delaunay
    tri = Delaunay(points[:, :2])  # Toma las coordenadas (x, y) para la triangulación
    
    # Calcular el número de nuevos puntos a crear
    num_original_points = len(points)
    num_new_points = int(num_original_points * (density_factor-1))

    # Calcular el número de puntos a generar por triángulo
    num_triangles = len(tri.simplices)
    points_per_triangle = num_new_points // num_triangles
    remaining_points = num_new_points % num_triangles
        
    new_points = []
    new_remissions = []
    
    for i, simplex in enumerate(tri.simplices):
        # Obtener los vértices del triángulo
        triangle_vertices = points[simplex] # Selecciona las coordenadas de los vertices que forma el triangulo
        triangle_remissions = remissions[simplex] # intensidades de los vertices del triángulo

        # Calcular el número de puntos para este triángulo
        # points_per_triangle es el número base de puntos a generar por triángulo
        # remaining_points es el resto de puntos que no se han distribuido uniformemente
        # Si el índice del triángulo (i) es menor que remaining_points, se genera un punto adiciona
        num_points_this_triangle = points_per_triangle + (1 if i < remaining_points else 0)
                
        # Generar nuevos puntos dentro del triangulo actual
        for _ in range(num_points_this_triangle):

            # Vértices del triángulo (V1, V2, V3)
            V1, V2, V3 = triangle_vertices
            # Generar coordenadas baricéntricas aleatorias
            w1, w2 = np.random.rand(2)
            if w1 + w2 > 1:
                w1, w2 = 1 - w1, 1 - w2
            w3 = 1 - w1 - w2

            # Calcular el punto en el plano del triángulo
            new_point = w1 * triangle_vertices[0] + w2 * triangle_vertices[1] + w3 * triangle_vertices[2]

            # Ahora que new_point está definido, podemos acceder a sus coordenadas
            Px, Py = new_point[0], new_point[1]  # Coordenadas x e y del nuevo punto

            # Denominador común para w1 y w2
            denominator = (V2[1] - V3[1]) * (V1[0] - V3[0]) + (V3[0] - V2[0]) * (V1[1] - V3[1])

            # Calcular w1 y w2 usando las fórmulas de coordenadas baricéntricas
            w1 = ((V2[1] - V3[1]) * (Px - V3[0]) + (V3[0] - V2[0]) * (Py - V3[1])) / denominator
            w2 = ((V3[1] - V1[1]) * (Px - V3[0]) + (V1[0] - V3[0]) * (Py - V3[1])) / denominator
            w3 = 1 - w1 - w2

            # Calcular el punto en el plano del triángulo
            new_point = w1* triangle_vertices[0] + w2* triangle_vertices[1] + w3 * triangle_vertices[2]
            
            # Calcular la intensidad usando interpolación ponderada por distancia inversa
            # Las coordenadas baricéntricas (r1, r2, w3) ya son ponderaciones normalizadas
            remission_interp = w1* triangle_remissions[0] + w2* triangle_remissions[1] + w3 * triangle_remissions[2]

            # Agregar el nuevo punto y su intensidad a la lista
            new_points.append(new_point)
            new_remissions.append(remission_interp)
    
    combined_points = np.vstack([points, new_points])
    combined_remissions = np.concatenate([remissions, new_remissions])

    # Retornar los puntos originales junto con los nuevos puntos
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
        pcd.points = o3d.utility.Vectow3dVector(points)
        
        if remissions is not None and len(remissions) > 0:
            # Normalizar intensidades a escala de 0-1 y asignarlas como colores
            colors = np.tile(remissions[:, None], (1, 3)) / np.max(remissions)
            pcd.colors = o3d.utility.Vectow3dVector(colors)
        
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
            points, remissions = densify_point_cloud(points, remissions, density_factor)
            
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

# ------ Interfaz de usuario -------

def main():
    # Solicitar directorio de entrada por terminal
    print("\n=== Densificación de nubes de puntos con TIN ===")
    input_directory = input("\nIntroduce la ruta del directorio de entrada: ").strip()
    
    # Validar directorio de entrada
    if not os.path.isdir(input_directory):
        print(f"\nError: El directorio '{input_directory}' no existe.")
        return

    # Solicitar factor de densificación con validación
    while True:
        try:
            density_factor = float(input("\nIntroduce el factor de densificación (3-10 recomendado, rango 1-20): "))
            if 1.0 <= density_factor <= 20.0:
                break
            else:
                print("Error: El valor debe estar entre 1.0 y 20.0")
        except ValueError:
            print("Error: Introduce un número válido.")

    # Crear directorio de salida automáticamente
    output_dir_name = f"densified_TIN_{density_factor}"
    output_directory = os.path.join(input_directory, output_dir_name)
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"\nDirectorio de salida creado: {output_directory}")
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