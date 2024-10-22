import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt

# Función para leer un archivo PLY
def leer_archivo_ply(ruta_archivo):
    try:
        print(f"[DEBUG] Leyendo archivo PLY: {ruta_archivo}")
        cloud = o3d.io.read_point_cloud(ruta_archivo)
        if cloud.is_empty():
            print("[ERROR] La nube de puntos está vacía. Verifica el archivo PLY.")
            return None, None
        puntos = np.asarray(cloud.points)

        z_values = puntos[:, 2]
        z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min())  # Normalización
        
        # Aplicar el colormap
        cmap = plt.get_cmap('viridis')
        colores = cmap(z_norm)[:, :3]  # Obtener colores del colormap y usar sólo RGB

        # Filtrar los puntos que son [0, 0, 0]
        mask = ~(np.all(puntos == [0, 0, 0], axis=1))
        puntos_filtrados = puntos[mask]
        colores_filtrados = colores[mask]
        print(f"[DEBUG] Puntos filtrados: {puntos_filtrados.shape[0]} puntos")
        return puntos_filtrados, colores_filtrados
    except Exception as e:
        print(f"[ERROR] No se pudo leer el archivo {ruta_archivo}: {e}")
        return None, None

# Función para visualizar la nube de puntos
def visualizar_nube_puntos(vis, puntos, colores, tamaño=1.0):
    if puntos is None or len(puntos) == 0:
        print("[Advertencia] La nube de puntos está vacía, omitiendo visualización.")
        return
    nube_puntos = o3d.geometry.PointCloud()
    nube_puntos.points = o3d.utility.Vector3dVector(puntos)
    if colores is not None and len(colores) == len(puntos):
        nube_puntos.colors = o3d.utility.Vector3dVector(colores)
    vis.clear_geometries()  # Limpiar geometrías anteriores
    vis.add_geometry(nube_puntos)
    
    # Ajustar el tamaño de los puntos
    opt = vis.get_render_option()
    opt.point_size = tamaño

    # # Establecer el fondo negro
    # opt.background_color = np.asarray([0, 0, 0])  # Negro


    # # Ajustar la posición de la cámara
    # ctr = vis.get_view_control()
    # ctr.set_lookat([0, 0, 0])  # Punto de enfoque
    # ctr.set_front([0, 0, -1])  # Dirección de la cámara
    # ctr.set_up([0, -1, 0])     # Vector 'up' de la cámara
    # ctr.set_zoom(0.8)          # Nivel de zoom

    vis.update_renderer()

# Función principal
def main(carpeta):
    archivos = sorted([f for f in os.listdir(carpeta) if f.endswith('.ply')])
    if not archivos:
        print("[ADVERTENCIA] No se encontraron archivos .ply en la carpeta.")
        return
    
    # Crear ventana de visualización
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    index_actual = 0  # Índice del archivo actual

    # Mostrar el primer archivo
    puntos, colores = leer_archivo_ply(os.path.join(carpeta, archivos[index_actual]))
    visualizar_nube_puntos(vis, puntos, colores, tamaño=2.0)

    # Callback para tecla derecha
    def tecla_derecha(vis):
        nonlocal index_actual
        if index_actual < len(archivos) - 1:
            index_actual += 1
            puntos, colores = leer_archivo_ply(os.path.join(carpeta, archivos[index_actual]))
            visualizar_nube_puntos(vis, puntos, colores, tamaño=2.0)
            print(f"Mostrando archivo: {archivos[index_actual]}")
        return False

    # Callback para tecla izquierda
    def tecla_izquierda(vis):
        nonlocal index_actual
        if index_actual > 0:
            index_actual -= 1
            puntos, colores = leer_archivo_ply(os.path.join(carpeta, archivos[index_actual]))
            visualizar_nube_puntos(vis, puntos, colores, tamaño=2.0)
            print(f"Mostrando archivo: {archivos[index_actual]}")
        return False

    # Callback para tecla escape
    def tecla_escape(vis):
        vis.destroy_window()
        return False

    # Asignar callbacks
    vis.register_key_callback(262, tecla_derecha) # Flecha derecha
    vis.register_key_callback(263, tecla_izquierda) # Flecha izquierda
    vis.register_key_callback(256, tecla_escape) # Esc

    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    carpeta = "/home/daniel/2024-tfg-daniel-borja/codigo/lidarTest"
    main(carpeta)
