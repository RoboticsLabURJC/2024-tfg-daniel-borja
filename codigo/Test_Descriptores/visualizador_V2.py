import os
import sys
import numpy as np
import open3d as o3d
from plyfile import PlyData
from calcularDescriptores import LidarPointCloudDescriptors

#--------VARIABLES GLOBALES---------

current_index = 0
files = []
point_size = 2.0

#--------LECTURA DE ARCHIVOS---------

def read_bin_file(file_path):
    if not file_path.endswith('.bin') or not os.path.exists(file_path):
        print(f"Error: .bin file not found at {file_path}")
        sys.exit(1)
    print(f"Reading: {file_path}")
    scan = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
    points, remissions = scan[:, 0:3], scan[:, 3]
    return points, remissions

def read_ply_file(file_path):
    if not file_path.endswith('.ply') or not os.path.exists(file_path):
        print(f"Error: .ply file not found at {file_path}")
        sys.exit(1)
    plydata = PlyData.read(file_path)
    print(file_path)
    x, y, z = plydata['vertex'].data['x'], plydata['vertex'].data['y'], plydata['vertex'].data['z']
    points = np.vstack((x, y, z)).T
    remissions = plydata['vertex'].data['intensity']
    return points, remissions

def load_path_files(path) -> list:
    def extract_sample_number(file_name, extension):
        parts = file_name.split("__" if extension == '.bin' else "-")
        num_str = ''.join(filter(str.isdigit, parts[1] if extension == '.bin' else parts[0]))
        return int(num_str) if num_str.isdigit() else 0

    # Filtrar archivos .bin y .ply
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.bin') or file.endswith('.ply')]
    if files:
        ext = os.path.splitext(files[0])[1]
        return sorted(files, key=lambda file: extract_sample_number(os.path.basename(file), ext))
    return []

#--------SEGMENTACION---------

def segmentar_por_planaridad(pcd, umbral_planaridad=0.2, k_vecinos=50, altura_min=-1.0, altura_max=0.5):
    """
    Segmenta la nube de puntos en función de la planaridad y el rango de altura.

    :param pcd: Nube de puntos de Open3D
    :param umbral_planaridad: Umbral para considerar puntos como planos
    :param k_vecinos: Número de vecinos para calcular la planaridad
    :param altura_min: Altura mínima para considerar puntos como suelo
    :param altura_max: Altura máxima para considerar puntos como suelo
    :return: Índices de puntos planos y no planos, y planaridades calculadas
    """
    # Calcular normales
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=k_vecinos))
    
    # Orientar normales
    pcd.orient_normals_consistent_tangent_plane(100)
    
    # Crear objeto de descriptores
    descriptores = LidarPointCloudDescriptors(pcd)
    
    # Calcular planaridad
    planaridades = descriptores.calcular_planaridad(k_vecinos=k_vecinos)
    
    # Filtrar los puntos por altura (Z)
    puntos = np.asarray(pcd.points)
    indices_altura = np.where((puntos[:, 2] >= altura_min) & (puntos[:, 2] <= altura_max))[0]
    
    # Identificar puntos planos
    indices_planos = np.intersect1d(indices_altura, np.where(planaridades > umbral_planaridad)[0])
    indices_no_planos = np.setdiff1d(indices_altura, indices_planos)
    
    return indices_planos, indices_no_planos, planaridades

def visualizar_segmentacion(vis, pcd, indices_planos, indices_no_planos):
    """
    Visualiza la segmentación coloreando puntos planos y no planos.

    :param vis: El visualizador de Open3D
    :param pcd: Nube de puntos de Open3D
    :param indices_planos: Índices de los puntos planos
    :param indices_no_planos: Índices de los puntos no planos
    """
    # Crear copia de la nube
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = pcd.points

    # Colorear puntos: planos (verde) y no planos (negro)
    colors = np.zeros((len(pcd.points), 3))  # Inicializar colores en negro
    colors[indices_planos] = [0, 1, 0]  # Verde para puntos planos
    colors[indices_no_planos] = [0, 0, 0]  # Negro para puntos no planos

    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    # Añadir geometría al visualizador
    vis.clear_geometries()
    vis.add_geometry(pcd_colored)

    # Ajustar el tamaño de los puntos
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # Tamaño de los puntos

    # Iniciar la visualización
    vis.update_geometry(pcd_colored)

#--------VISUALIZACION---------

def downsample_point_cloud(pcd, voxel_size=0.1):
    """
    Reduce el número de puntos en la nube de puntos usando un muestreo por vóxeles.

    :param pcd: Nube de puntos de Open3D
    :param voxel_size: Tamaño del vóxel para la reducción
    :return: Nube de puntos reducida
    """
    return pcd.voxel_down_sample(voxel_size)

def visualizar_archivo(vis):
    global current_index, files, point_size
    
    if current_index >= len(files):  # Si estamos al final, detener
        print("No hay más archivos.")
        return False
    
    archivo = files[current_index]
    print(f"Mostrando archivo: {archivo}")
    
    # Leer archivo dependiendo de la extensión
    if archivo.endswith('.bin'):
        points, remissions = read_bin_file(archivo)
    elif archivo.endswith('.ply'):
        points, remissions = read_ply_file(archivo)
    else:
        print("Formato de archivo no soportado.")
        return False
    
    # Crear la nube de puntos
    pcd = o3d.geometry.PointCloud()
    
    # Reducir el tamaño de la nube de puntos
    pcd = downsample_point_cloud(pcd, voxel_size=0.05)  
    
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Parámetros de segmentación
    umbral_planaridad = 0.2
    k_vicinos = 50
    altura_min = -1.0  # Establecer un valor adecuado de altura mínima
    altura_max = 0.5   # Establecer un valor adecuado de altura máxima
    
    # Segmentar por planaridad
    indices_planos, indices_no_planos, planaridades = segmentar_por_planaridad(pcd, umbral_planaridad, k_vicinos, altura_min, altura_max)

    # Visualizar resultados
    print(f"Puntos planos: {len(indices_planos)}, Puntos no planos: {len(indices_no_planos)}")
    
    # Asegúrate de pasar los tres argumentos a visualizar_segmentacion
    visualizar_segmentacion(vis, pcd, indices_planos, indices_no_planos)

    
#--------NAVEGAR---------

def avanzar_archivo(vis):
    global current_index, files
    if current_index < len(files) - 1:
        current_index += 1  # Avanzar al siguiente archivo
        visualizar_archivo(vis)
    else:
        print("Último archivo alcanzado. No hay más archivos para mostrar.")

def retroceder_archivo(vis):
    global current_index, files
    if current_index > 0:
        current_index -= 1  # Retroceder al archivo anterior
        visualizar_archivo(vis)
    else:
        print("Primer archivo alcanzado. No se puede retroceder.")

#--------MAIN---------

def main():
    global files, current_index
    
    # Path a los archivos
    #path = '/home/dani/2024-tfg-daniel-borja/datasets/goose_3d_val/lidar/val/2022-08-30_siegertsbrunn_feldwege/'
    path = "/home/dani/2024-tfg-daniel-borja/datasets/muestras"
    files = load_path_files(path)
    
    if not files:
        print("No se encontraron archivos en el directorio especificado.")
        return
    
    # Inicializar visualización
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    # Mostrar el primer archivo
    visualizar_archivo(vis)
    
    # Asociar eventos de teclado
    vis.register_key_callback(262, avanzar_archivo)  # Flecha derecha
    vis.register_key_callback(263, retroceder_archivo)  # Flecha izquierda
    
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
