import numpy as np
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
import sys
import tkinter as tk
from tkinter import filedialog, simpledialog
from sklearn.neighbors import NearestNeighbors

# ------  Lectura de archivos -------

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

# ------ Reducción por Voxelización -------

def calculate_voxel_size(points: np.ndarray, density_index: float) -> float:
    """
    Calcula el tamaño del voxel necesario para lograr un índice de densidad dado.
    
    :param points: Nube de puntos original (N, 3).
    :param density_index: Índice de densidad deseado (0 < density_index <= 1).
    :return: Tamaño del voxel calculado.
    """
    if density_index <= 0 or density_index > 1:
        raise ValueError("El índice de densidad debe estar entre 0 y 1.")
    
    # Calcular el volumen del bounding box de la nube de puntos
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    bounding_box_size = max_bound - min_bound
    volume = np.prod(bounding_box_size)  # Volumen del bounding box
    
    # Calcular el tamaño del voxel
    num_points = len(points)
    voxel_size = (volume / (density_index * num_points)) ** (1/3)
    
    return voxel_size

def reduce_point_cloud_by_voxel(points: np.ndarray, remissions: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce la densidad de la nube de puntos utilizando voxelización.
    
    :param points: Nube de puntos original (N, 3)
    :param remissions: Intensidades de los puntos originales (N,)
    :param voxel_size: Tamaño del voxel (en las mismas unidades que las coordenadas de los puntos)
    :return: Puntos reducidos y sus intensidades
    """
    # Crear una nube de puntos en Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Aplicar filtrado por voxelización
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    
    # Obtener los puntos reducidos
    reduced_points = np.asarray(downsampled_pcd.points)
    
    # Si hay intensidades, asignarlas a los puntos reducidos
    if len(remissions) > 0:
        # Encontrar los índices de los puntos originales más cercanos a los puntos reducidos
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(points)
        _, indices = nn.kneighbors(reduced_points)
        reduced_remissions = remissions[indices.flatten()]
    else:
        reduced_remissions = np.zeros(len(reduced_points))
    
    return reduced_points, reduced_remissions

# ------ Guardado de archivos -------

def save_point_cloud(points: np.ndarray, remissions: np.ndarray, output_path: str, input_extension: str):
    """
    Guarda la nube de puntos reducida en el formato especificado.
    
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

def batch_reduction_by_density(input_directory: str, output_directory: str, density_index: float):
    """
    Procesa todos los archivos en un directorio y reduce la densidad de las nubes de puntos por índice de densidad.
    
    :param input_directory: Directorio de entrada con archivos .bin, .ply o .pcd.
    :param output_directory: Directorio de salida para guardar los archivos reducidos.
    :param density_index: Índice de densidad para la reducción (0 < density_index <= 1).
    """
    os.makedirs(output_directory, exist_ok=True)
    
    input_files = [f for f in os.listdir(input_directory) 
                   if f.endswith(('.bin', '.ply', '.pcd'))]
    
    for file in input_files:
        input_path = os.path.join(input_directory, file)
        output_filename = f"reduced_{file}"  # Mantener misma extensión
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
            
            # Calcular el tamaño del voxel necesario para lograr el índice de densidad
            voxel_size = calculate_voxel_size(points, density_index)
            print(f"Tamaño del voxel calculado: {voxel_size}")
            
            # Reducir la densidad de la nube de puntos por voxelización
            reduced_points, reduced_remissions = reduce_point_cloud_by_voxel(
                points, remissions, voxel_size
            )
            
            print(f"Número de puntos reducido: {len(reduced_points)}")
            
            # Guardar los puntos reducidos en el mismo formato
            save_point_cloud(
                reduced_points, 
                reduced_remissions, 
                output_path,
                input_extension
            )
        
        except Exception as e:
            print(f"Error procesando {file}: {e}")

def main():
    root = tk.Tk()
    root.withdraw()  # Ocultar ventana principal

    # Seleccionar directorio de entrada
    input_directory = filedialog.askdirectory(title="Seleccionar directorio de entrada")
    if not input_directory:
        print("No se seleccionó ningún directorio.")
        return

    # Seleccionar directorio de salida
    output_directory = filedialog.askdirectory(title="Seleccionar directorio de salida")
    if not output_directory:
        print("No se seleccionó ningún directorio de salida.")
        return

    # Solicitar índice de densidad
    density_index = simpledialog.askfloat(
        "Índice de Densidad", 
        "Introduce el índice de densidad para la reducción (por ejemplo, 0.5 para reducir a la mitad):", 
        initialvalue=1.0, minvalue=0.01, maxvalue=1.0
    )

    if density_index is not None:
        batch_reduction_by_density(
            input_directory, 
            output_directory, 
            density_index
        )
        print("Proceso de reducción por índice de densidad completado.")

if __name__ == "__main__":
    main()