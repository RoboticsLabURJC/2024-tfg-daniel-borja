import numpy as np
import open3d as o3d
from calcularDescriptores import LidarPointCloudDescriptors

def segmentar_por_planaridad(pcd, umbral_planaridad, k_vecinos, altura_min, altura_max):
    """
    Segmenta la nube de puntos en función de la planaridad.

    :param pcd: Nube de puntos de Open3D
    :param umbral_planaridad: Umbral para considerar puntos como planos
    :param k_vecinos: Número de vecinos para calcular la planaridad
    :return: Índices de puntos planos y no planos
    """
    # Crear objeto de descriptores
    descriptores = LidarPointCloudDescriptors(pcd)
    
    # Calcular planaridad
    planaridades = descriptores.calcular_planaridad(k_vecinos=k_vecinos)
    
    # Filtrar puntos según la planaridad
    indices_planos = np.where(planaridades > umbral_planaridad)[0]
    
    # Obtener las coordenadas Z de todos los puntos
    alturas_z = np.asarray(pcd.points)[:, 2]
    print(f"Altura maxima: {np.max(alturas_z):.4f}")
    print(f"Altura minima: {np.min(alturas_z):.4f}")

    # Filtrar puntos por el rango de altura (altura_min <= Z <= altura_max)
    indices_suelo_filtrados = indices_planos[(alturas_z[indices_planos] >= altura_min) & (alturas_z[indices_planos] <= altura_max)]
    
    # Los puntos que no están en el rango de altura o no son planos se consideran no planos
    indices_no_planos = np.setdiff1d(np.arange(len(pcd.points)), indices_suelo_filtrados)
    
    return indices_suelo_filtrados, indices_no_planos, planaridades

def visualizar_segmentacion(pcd, indices_planos, indices_no_planos):
    """
    Visualiza la segmentación coloreando puntos planos y no planos.

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
    
    # Cargar la nube de puntos
    pcd = o3d.io.read_point_cloud(archivo)
    
    # Segmentar por planaridad
    umbral_planaridad = 0.6 # Ajusta según lo necesario
    k_vecinos = 90
    altura_min = -10.0
    altura_max = -0.6

    # Segmentar por planaridad y filtrar por altura
    indices_suelo_filtrados, indices_no_planos, planaridades = segmentar_por_planaridad(pcd, umbral_planaridad, k_vecinos, altura_min, altura_max)

# Visualizar resultados
    print(f"Puntos de suelo filtrados: {len(indices_suelo_filtrados)}, Puntos no planos: {len(indices_no_planos)}")
    visualizar_segmentacion(pcd, indices_suelo_filtrados, indices_no_planos)
   
    # Imprimir estadísticas de planaridad
    print(f"Planaridad mínima: {np.min(planaridades):.4f}")
    print(f"Planaridad máxima: {np.max(planaridades):.4f}")
    print(f"Planaridad media: {np.mean(planaridades):.4f}")
    print(f"Planaridad percentil 90: {np.percentile(planaridades, 90):.4f}")


if __name__ == "__main__":
    main()
