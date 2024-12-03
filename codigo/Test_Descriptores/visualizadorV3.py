import numpy as np
import open3d as o3d
import os
import sys
from plyfile import PlyData
from calcularDescriptores import LidarPointCloudDescriptors

def cargar_nube_de_puntos(archivo):
    """
    Carga la nube de puntos según el tipo de archivo.

    :param archivo: Ruta del archivo de nube de puntos
    :return: Nube de puntos de Open3D
    """
    # Verificar que el archivo exista
    if not os.path.exists(archivo):
        print(f"Error: Archivo no encontrado en {archivo}")
        sys.exit(1)

    # Crear nube de puntos de Open3D
    pcd = o3d.geometry.PointCloud()

    # Procesar según la extensión del archivo
    if archivo.lower().endswith('.bin'):
        # Leer archivo .bin (típicamente formato KITTI)
        scan = np.fromfile(archivo, dtype=np.float32)
        points = scan.reshape((-1, 4))[:, :3]  # x, y, z
        pcd.points = o3d.utility.Vector3dVector(points)
        
        print(f"Cargado archivo .bin: {archivo}")
        print(f"Número de puntos: {len(points)}")

    elif archivo.lower().endswith('.ply'):
        # Leer archivo .ply usando Open3D directamente
        pcd = o3d.io.read_point_cloud(archivo)
        
        print(f"Cargado archivo .ply: {archivo}")
        print(f"Número de puntos: {len(pcd.points)}")

    else:
        print(f"Formato de archivo no soportado: {archivo}")
        sys.exit(1)

    return pcd

def segmentar_punto_nube_multi_clase(pcd, 
                                     umbral_planaridad, 
                                     k_vecinos, 
                                     altura_min, 
                                     altura_max,
                                     umbral_variacion_superficial,
                                     umbral_anisotropia):
    """
    Segmenta la nube de puntos en múltiples clases.

    :param pcd: Nube de puntos de Open3D
    :param umbral_planaridad: Umbral para considerar puntos como planos
    :param k_vecinos: Número de vecinos para calcular descriptores
    :param altura_min: Altura mínima para considerar puntos de suelo
    :param altura_max: Altura máxima para considerar puntos de suelo
    :param umbral_variacion_superficial: Umbral para identificar árboles
    :return: Índices de puntos de suelo, árboles y otros
    """
    # Crear objeto de descriptores
    descriptores = LidarPointCloudDescriptors(pcd)
    
    # Calcular planaridad
    planaridades = descriptores.calcular_planaridad(k_vecinos=k_vecinos)
    
    # Calcular variación superficial
    variacion_superficial = descriptores.calcular_variacion_superficial(k_vecinos=k_vecinos)
    
    #Calcular anisotropia
    anisotropia = descriptores.calcular_anisotropia(k_vecinos=k_vecinos)

    # Obtener las coordenadas Z de todos los puntos
    alturas_z = np.asarray(pcd.points)[:, 2]
    
    # Filtrar puntos de suelo
    indices_planos = np.where(planaridades > umbral_planaridad)[0]
    indices_suelo_filtrados = indices_planos[
        (alturas_z[indices_planos] >= altura_min) & 
        (alturas_z[indices_planos] <= altura_max)
    ]
    
    # Identificar puntos de árboles (alta variación superficial)
    indices_arboles = np.where(
        (variacion_superficial > umbral_variacion_superficial) & 
        (alturas_z > altura_max)  # Puntos por encima de la altura del suelo
    )[0]

    # Identificar puntos de personas
    indices_personas = np.where(
        (variacion_superficial < umbral_variacion_superficial) & 
        (planaridades > umbral_planaridad) &
        (anisotropia < umbral_anisotropia)
    )[0]

    # Identificar puntos que son otra cosa
    indices_otros = np.setdiff1d(
        np.arange(len(pcd.points)), 
        np.concatenate([indices_suelo_filtrados, indices_arboles, indices_personas])
    )
    
    # Información de depuración
    print(f"Puntos de suelo: {len(indices_suelo_filtrados)}")
    print(f"Puntos de árboles: {len(indices_arboles)}")
    print(f"Puntos de personas: {len(indices_personas)}")
    print(f"Otros puntos: {len(indices_otros)}")
    print(f"Variación superficial - Mín: {np.min(variacion_superficial):.4f}")
    print(f"Variación superficial - Máx: {np.max(variacion_superficial):.4f}")
    print(f"Variación superficial - Media: {np.mean(variacion_superficial):.4f}")
    
    return indices_suelo_filtrados, indices_arboles, indices_personas, indices_otros, variacion_superficial

def visualizar_segmentacion(pcd, indices_suelo, indices_arboles, indices_personas, indices_otros):
    """
    Visualiza la segmentación coloreando puntos por clase.

    :param pcd: Nube de puntos de Open3D
    :param indices_suelo: Índices de puntos de suelo
    :param indices_arboles: Índices de puntos de árboles
    :param indices_personas: Indices de puntos de personas
    :param indices_otros: Índices de otros puntos
    """
    # Crear copia de la nube
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = pcd.points

    # Colorear puntos: planos (verde) y no planos (negro)
    colors = np.zeros((len(pcd.points), 3))  # Inicializar colores en negro
    colors[indices_suelo] = [1, 0.7, 0.2]  # Verde para suelo
    colors[indices_arboles] = [0, 0.7, 0] # Rojo para arboles
    colors[indices_personas] = [0, 0, 0.7] # Azul para personas
    colors[indices_otros] = [0, 0, 0]  # Negro para otros

    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    # Configurar el visualizador
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_colored)

    # Ajustar el tamaño de los puntos
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # Tamaño de los puntos

    # Iniciar la visualización
    vis.run()
    vis.destroy_window()

def main():
    # Ruta al archivo de nube de puntos
    archivo = "/home/dani/2024-tfg-daniel-borja/datasets/Rellis_3D_os1_cloud_node_color_ply/Rellis-3D/00000/os1_cloud_node_color_ply/frame000000-1581624652_770.ply"  # Cambia por tu archivo
    #archivo = "/home/dani/2024-tfg-daniel-borja/datasets/goose_3d_val/lidar/val/2022-07-22_flight/2022-07-22_flight__0071_1658494234334310308_vls128.bin"  # Cambia por tu archivo

    # Cargar la nube de puntos
    pcd = o3d.io.read_point_cloud(archivo)
    
    # Segmentar por planaridad
    umbral_planaridad = 0.6 # Ajusta según lo necesario
    k_vecinos = 90
    altura_min = -10.0
    altura_max = -0.6
    umbral_variacion_superficial = 0.1
    umbral_anisotropia = 0.1

    # Segmentar nube de puntos
    indices_suelo, indices_arboles, indices_personas, indices_otros, variacion_superficial = segmentar_punto_nube_multi_clase(
        pcd, 
        umbral_planaridad, 
        k_vecinos, 
        altura_min, 
        altura_max,
        umbral_variacion_superficial,
        umbral_anisotropia  # Añade este parámetro con un valor por defecto
    )

    # Visualizar resultados
    visualizar_segmentacion(pcd, indices_suelo, indices_arboles, indices_personas, indices_otros)

    # Imprimir estadísticas de planaridad
    # print(f"Planaridad mínima: {np.min(planaridades):.4f}")
    # print(f"Planaridad máxima: {np.max(planaridades):.4f}")
    # print(f"Planaridad media: {np.mean(planaridades):.4f}")
    # print(f"Planaridad percentil 90: {np.percentile(planaridades, 90):.4f}")


if __name__ == "__main__":
    main()
