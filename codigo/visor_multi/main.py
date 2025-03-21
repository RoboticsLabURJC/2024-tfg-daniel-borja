import open3d as o3d
import numpy as np
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox

# ------ Parámetros y valores por defecto -------
FPS = 10  # Fotogramas por segundo

# ------- Funciones de Visualización -------
def read_bin_file(file_path):
    """Lee un archivo .bin y devuelve los puntos y las intensidades usando NumPy."""
    if not file_path.endswith('.bin') or not os.path.exists(file_path):
        print(f"Error: .bin file not found at {file_path}")
        return None, None
    scan = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))  # Leer archivo .bin
    points = scan[:, 0:3]  # Coordenadas x, y, z
    intensities = scan[:, 3]  # Intensidades
    return points, intensities

def load_path_files(path):
    """Carga todos los archivos .bin de un directorio y los ordena."""
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.bin')]
    return sorted(files)

def visualize_4_directories(directory_paths):
    """Visualiza nubes de puntos de 4 directorios en una pantalla dividida."""
    # Cargar archivos de los 4 directorios
    file_lists = [load_path_files(path) for path in directory_paths]
    num_frames = min(len(files) for files in file_lists)  # Número de fotogramas comunes

    # Crear 4 nubes de puntos
    point_clouds = [o3d.geometry.PointCloud() for _ in range(4)]

    # Crear una ventana de Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='4 Directorios', width=1920, height=1080)

    # Añadir las nubes de puntos a la ventana
    for i, point_cloud in enumerate(point_clouds):
        vis.add_geometry(point_cloud)

    # Configurar la cámara
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, 1])  # Orientación de la cámara
    view_control.set_lookat([0, 0, 0])  # Punto de enfoque
    view_control.set_up([0, 1, 0])  # Vector "arriba"
    view_control.set_zoom(0.1)  # Zoom inicial

    frame = 0
    while True:
        for i in range(4):  # Actualizar cada nube de puntos
            file_path = file_lists[i][frame % num_frames]
            points, intensities = read_bin_file(file_path)  # Leer archivo .bin

            if points is None or intensities is None:
                print(f"Error al cargar el archivo: {file_path}")
                continue

            # Crear una nube de puntos con Open3D usando los datos de NumPy
            point_clouds[i].points = o3d.utility.Vector3dVector(points)  # Actualizar puntos
            colors = np.zeros((points.shape[0], 3))  # Colores basados en intensidades
            colors[:, 0] = intensities  # Usar intensidades para el canal rojo
            point_clouds[i].colors = o3d.utility.Vector3dVector(colors)  # Actualizar colores

            vis.update_geometry(point_clouds[i])  # Actualizar visualización

        vis.poll_events()  # Procesar eventos
        vis.update_renderer()  # Actualizar renderizado
        time.sleep(1 / FPS)  # Controlar la velocidad de visualización
        frame += 1

# ------- Interfaz gráfica para seleccionar directorios -------
def select_directories():
    """Selecciona 4 directorios usando una ventana de Tkinter."""
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter

    directory_paths = []
    for i in range(4):
        path = filedialog.askdirectory(title=f"Selecciona el directorio {i + 1}")
        if not path:
            messagebox.showerror("Error", "Debes seleccionar 4 directorios.")
            return None
        directory_paths.append(path)

    return directory_paths

# ------ Programa Principal ------
if __name__ == "__main__":
    directory_paths = select_directories()  # Seleccionar directorios
    if directory_paths:
        visualize_4_directories(directory_paths)  # Visualizar nubes de puntos