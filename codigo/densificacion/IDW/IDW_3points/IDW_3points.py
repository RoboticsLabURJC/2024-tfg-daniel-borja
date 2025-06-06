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
                                       density_factor: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Densifica la nube de puntos utilizando interpolación basada en los dos vecinos más cercanos.
    """
    # Usar NearestNeighbors para encontrar los vecinos más cercanos
    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(points)
    
    # Establecer el número de nuevos puntos a crear
    num_new_points = int((len(points) * (density_factor-1)))
    
    # Cuenta de las veces que se usa cada punto
    point_usage_count = {}

    # Almacenar los puntos y las intensidades generadas
    new_points = []
    new_remissions = []

    #Parámetros de la distribución normal truncada
    media = 0.5
    desviacion_estandar = 0.2
    limite_inferior = 0
    limite_superior = 1   

    #Convertir los límites al espacio de la distribucion normal   
    a = (limite_inferior - media) / desviacion_estandar
    b = (limite_superior - media) / desviacion_estandar
    
    for _ in range(num_new_points):
        # Elegir un punto aleatorio de la nube de puntos original
        index = np.random.randint(0, len(points))
        point = points[index]

        # Elegir un punto que haya sido usado menos veces
        if not point_usage_count:  # Si el diccionario está vacío
            index = np.random.randint(0, len(points))

        else:
            # Encontrar el mínimo número de veces que se ha usado cualquier punto
            min_usage = min(point_usage_count.values())
            # Obtener todos los índices que tienen el mínimo uso
            least_used_indices = [idx for idx, count in point_usage_count.items() if count == min_usage]
            # Si hay puntos que aún no se han usado, incluirlos
            unused_indices = [i for i in range(len(points)) if i not in point_usage_count]
            candidate_indices = least_used_indices + unused_indices
            # Seleccionar aleatoriamente entre los candidatos
            index = np.random.choice(candidate_indices)

        point = points[index]

        # Incrementar contador para el punto seleccionado
        if index in point_usage_count:
            point_usage_count[index] += 1
        else:
            point_usage_count[index] = 1

        
        # Obtener los 2 vecinos más cercanos (el primero es él mismo)
        distances,indices = nn.kneighbors([point], n_neighbors=3)

        # Seleccionar los puntos para la interpolacion
        point_A = points[indices[0][0]]
        point_B = points[indices[0][1]]
        point_C = points[indices[0][2]]

        # Generar pesos aleatorios que sumen 1
        lambda_1 = truncnorm.rvs(a ,b, loc=media, scale=desviacion_estandar)
        lambda_2 = truncnorm.rvs(a ,b, loc=media, scale=desviacion_estandar)
        lambda_3 = truncnorm.rvs(a ,b, loc=media, scale=desviacion_estandar)

         # Normalizar los pesos para que sumen 1
        total = lambda_1 + lambda_2 + lambda_3
        lambda_1 /= total
        lambda_2 /= total
        lambda_3 /= total
                
        # Interpolacion aplicando D=λ1*​A+λ2*​B+λ3*​C
        interpolated_point = lambda_1 * point_A + lambda_2 * point_B + lambda_3 * point_C

        # Calcular las distancias d_A, d_B y d_C
        d_A = np.linalg.norm(interpolated_point - point_A)  # Distancia al punto original
        d_B = np.linalg.norm(interpolated_point - point_B)  # Distancia al primer vecino
        d_C = np.linalg.norm(interpolated_point - point_C)  # Distancia al segundo vecino

        # Aplicar la fórmula de interpolación para la remisión
        r_A = remissions[index]  # Remisión del punto original
        r_B = remissions[indices[0][1]]  # Remisión del primer vecino
        r_C = remissions[indices[0][2]]  # Remisión del segundo vecino

        # Fórmula de interpolación de la remisión
        interpolated_remission = ( (1/d_A) * r_A + (1/d_B) * r_B + (1/d_C) * r_C ) / ( (1/d_A) + (1/d_B) + (1/d_C) )
        
        # Agregar el nuevo punto y su intensidad
        new_points.append(interpolated_point)
        new_remissions.append(interpolated_remission)
 
    # Mostrar el uso de los puntos
    print("\nInformación de puntos usados como referencia:")
    print("Formato: [Punto ID] (x, y, z) - usado N veces - Tipo")
    for point_idx, count in point_usage_count.items():
        coord = points[point_idx]
        print(f"[{point_idx}] ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}) - usado {count} veces - ORIGINAL")

    # También podemos mostrar los nuevos puntos generados
    print("\nPuntos nuevos generados:")
    for i, new_point in enumerate(new_points):
        print(f"[NEW_{i}] ({new_point[0]:.3f}, {new_point[1]:.3f}, {new_point[2]:.3f}) - INTERPOLADO")

    # Convertir las listas a arrays
    new_points = np.array(new_points)
    new_remissions = np.array(new_remissions)

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
