import numpy as np
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
import sys
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import filedialog, simpledialog

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

def densify_point_cloud(points: np.ndarray, remissions: np.ndarray, 
                                       density_factor: float = 5.0, n_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Densifica la nube de puntos utilizando interpolación basada en vecinos más cercanos.
    
    :param points: Nube de puntos original (N, 3)
    :param remissions: Intensidades de los puntos originales (N,)
    :param density_factor: Factor de densificación (número de nuevos puntos a agregar)
    :param n_neighbors: Número de vecinos cercanos para interpolación
    :return: Nuevos puntos generados, sus intensidades, y los puntos originales
    """
    # Usar NearestNeighbors para encontrar los vecinos más cercanos
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(points)
    
    # Establecer el número de nuevos puntos a crear
    num_new_points = int(len(points) * density_factor)
    
    # Almacenar los puntos y las intensidades generadas
    new_points = []
    new_remissions = []
    
    for _ in range(num_new_points):
        # Elegir un punto aleatorio de la nube de puntos original
        index = np.random.randint(0, len(points))
        point = points[index]
        
        # Obtener los vecinos más cercanos
        distances, indices = nn.kneighbors([point], n_neighbors=n_neighbors)
        
        # Seleccionar un vecino aleatorio entre los más cercanos para interpolar
        neighbor_idx = np.random.choice(indices[0])
        neighbor_point = points[neighbor_idx]
        
        # Interpolar entre el punto original y el vecino seleccionado
        interpolated_point = (point + neighbor_point) / 2  # Promedio de la posición
        
        # Las intensidades de los nuevos puntos serán las medias de las intensidades de los puntos vecinos
        interpolated_remission = np.mean(remissions[indices[0]])
        
        # Agregar el nuevo punto y su intensidad
        new_points.append(interpolated_point)
        new_remissions.append(interpolated_remission)
    
    # Convertir las listas a arrays
    new_points = np.array(new_points)
    new_remissions = np.array(new_remissions)
    
    # Retornar los puntos originales junto con los nuevos puntos
    return points, remissions, new_points, new_remissions


def save_point_cloud(points: np.ndarray, remissions: np.ndarray, output_path: str):
    """Guarda la nube de puntos densificada en formato PLY."""
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
    print(f"Archivo densificado guardado en: {output_path}")

def batch_densification_with_viz(input_directory: str, output_directory: str, 
                                  density_factor: float = 5.0):
    
    """Procesa archivos .bin y .ply en lotes, mostrando visualización y guardando resultados."""
    os.makedirs(output_directory, exist_ok=True)
    
    input_files = [f for f in os.listdir(input_directory) 
                   if f.endswith(('.bin', '.ply'))]
    
    for file in input_files:
        input_path = os.path.join(input_directory, file)
        output_filename = f"densified_{file.replace('.bin', '.ply').replace('.ply', '.ply')}"
        output_path = os.path.join(output_directory, output_filename)
        
        try:
            # Leer archivo dependiendo de su extensión
            if file.endswith('.bin'):
                points, remissions = read_bin_file(input_path)
            elif file.endswith('.ply'):
                points, remissions = read_ply_file(input_path)
            
            # Mostrar número de puntos originales
            print(f"Número de puntos original: {len(points)}")
            
            # Densificar la nube de puntos
            points, remissions, noise_points, noise_remissions = densify_point_cloud(
                points, remissions, density_factor
            )
            
            # Mostrar número de puntos densificados
            print(f"Número de puntos densificado: {len(points) + len(noise_points)}")
            
            # Guardar los puntos densificados
            save_point_cloud(
                np.vstack([points, noise_points]), 
                np.concatenate([remissions, noise_remissions]), 
                output_path
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

    # Solicitar factor de densificación
    density_factor = simpledialog.askfloat(
        "Factor de Densificación", 
        "Introduce el factor de densificación (recomendado: 3-10):", 
        initialvalue=5.0, minvalue=1.0, maxvalue=20.0
    )

    if density_factor is not None:
        batch_densification_with_viz(
            input_directory, 
            output_directory, 
            density_factor, 
        )
        print("Proceso de densificación completado.")

if __name__ == "__main__":
    main()

