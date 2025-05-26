import numpy as np
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
import sys
from typing import Tuple
from scipy.spatial import Delaunay
from tqdm import tqdm 
import time

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
                        density_factor: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Densifica la nube de puntos utilizando triangulación de Delaunay e interpolación lineal.
    """
    print("\n=== INICIO DEL PROCESO DE DENSIFICACION MEDIANTE TIN ===")
    print(f"Puntos iniciales: {points.shape}, Remisiones: {remissions.shape}")

    # Crear la triangulación de Delaunay
    tri = Delaunay(points[:, :2])  # Toma las coordenadas (x, y) para la triangulación
    print(f"✅ Triangulación completada. Número de triángulos: {len(tri.simplices)}")

    # Calcular el número de nuevos puntos a crear
    num_original_points = len(points)
    num_new_points = int(num_original_points * (density_factor-1))
    print(f"\n1. Generando {num_new_points} nuevos puntos...")

    # Calcular el número de puntos a generar por triángulo
    print("\n3. Distribuyendo puntos entre los triángulos...")
    num_triangles = len(tri.simplices)

    # Distribuir puntos por los trangulos
    base_points_per_tri = num_new_points // num_triangles
    extra_points = num_new_points % num_triangles

    # Crear máscara para triángulos que reciben punto extra
    print(f" - Puntos base por triángulo: {base_points_per_tri}")
    print(f" - Triángulos con punto extra: {extra_points}")
    extra_mask = np.zeros(num_triangles, dtype=bool)
    extra_mask[:extra_points] = True
        
    # Vector de conteo de puntos por triángulo
    points_per_triangle = np.full(num_triangles, base_points_per_tri)
    points_per_triangle[extra_mask] += 1

    # Generar coordenadas baricéntricas para todos los puntos
    print("\n4. Generando coordenadas baricéntricas aleatorias...")
    total_points = np.sum(points_per_triangle)
    print(f" - Total puntos a generar: {total_points}")

    # Generar coordenadas aleatorias
    w1_w2 = np.random.rand(total_points, 2)
    mask = w1_w2.sum(axis=1) > 1
    w1_w2[mask] = 1 - w1_w2[mask]
    w3 = 1 - w1_w2.sum(axis=1)
    barycentric = np.column_stack((w1_w2, w3))

    print("\nCoordenadas baricéntricas (primeras 10):")
    print(barycentric[:10])


    # Asignar triángulos a cada punto nuevo
    print("\n5. Asignando triángulos a los puntos...")
    triangle_indices = np.repeat(np.arange(num_triangles), points_per_triangle)
    print("\nÍndices de triángulos asignados (primeros 10):")
    print(triangle_indices[:10])

    # Obtener vértices para cada punto nuevo
    print("\n6. Obteniendo vértices para interpolación...")
    vertices = points[tri.simplices[triangle_indices]]
    vertex_remissions = remissions[tri.simplices[triangle_indices]]
    print("\nVértices de triángulos (primeros 10):")
    print(vertices[:10])

    # Interpolación vectorizada
    print("\n7. Realizando interpolación...")
    new_points = np.sum(barycentric[:, :, None] * vertices, axis=1)
    new_remissions = np.sum(barycentric * vertex_remissions, axis=1)

    print("\nPuntos interpolados (primeros 10):")
    print(new_points[:10])
    print("\nRemisiones interpoladas (primeras 10):")
    print(new_remissions[:10])

    # Combinar resultados
    print("\n8. Combinando con puntos originales...")
    combined_points = np.vstack([points, new_points])
    combined_remissions = np.concatenate([remissions, new_remissions])

    print("\n=== RESULTADOS FINALES ===")
    print(f"Total puntos originales: {len(points)}")
    print(f"Total puntos nuevos: {len(new_points)}")
    print(f"Total puntos combinados: {len(combined_points)}")
    print("\nPrimeros 5 puntos originales:")
    print(points[:5])
    print("\nÚltimos 5 puntos generados:")
    print(new_points[-5:])
    
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

def batch_densification(input_dir: str, density_factor: float = 5.0):
     # Crear nombre de carpeta basado en el factor de densificación
    densified_folder_name = f"densified_TIN_{density_factor}"
    output_dir = os.path.join(input_dir, densified_folder_name)

    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n Los archivos densificados se guardarán en: {output_dir}")

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
        output_path = os.path.join(output_dir, f"densified_{file}")
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
            
            # Densificar
            start_time = time.time()
            dense_points, dense_remissions = densify_point_cloud(points, remissions, density_factor)
            elapsed = time.time() - start_time
            
            print(f"✅ Densificado a {len(dense_points)} puntos (took {elapsed:.2f}s)")
            
            # Guardar
            save_point_cloud(dense_points, dense_remissions, output_path, ext)
            print(f"💾 Guardado como: densified_{file}")
            
        except Exception as e:
            print(f"\n❌ Error procesando {file}: {str(e)}")
    
    print("\n Proceso completado!")

# ------ Interfaz de usuario -------

def main():

    print("\n" + "="*50)
    print(" DENSIFICADOR DE NUBES DE PUNTOS LIDAR")
    print("="*50 + "\n")

    # Solicitar directorio de entrada
    while True:
        input_dir = input("📂 Introduzca la ruta del directorio de entrada: ").strip()
        if os.path.isdir(input_dir):
            break
        print("⚠ El directorio no existe. Intente nuevamente.")

    # Solicitar factor de densificación
    while True:
        try:
            density_factor = float(input("\n🔢 Introduzca el factor de densificación (3-10 recomendado): "))
            if 1.0 <= density_factor <= 20.0:
                break
            print("⚠ El valor debe estar entre 1.0 y 20.0")
        except ValueError:
            print("⚠ Debe introducir un número válido")


    # Confirmación
    print(f"\n⚙ Configuración:")
    print(f" - Directorio de entrada: {input_dir}")
    print(f" - Factor de densificación: {density_factor}")
    print(f" - Los resultados se guardarán en: {os.path.join(input_dir, f'densified_IDW_2points_{density_factor}')}")
    
    batch_densification(input_dir, density_factor)

if __name__ == "__main__":
    main()