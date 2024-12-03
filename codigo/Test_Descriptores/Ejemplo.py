from calcularDescriptores import LidarPointCloudDescriptors

def main():
    # Ruta a tu archivo de nube de puntos LiDAR (.ply o .bin)
    ruta_punto_nube = '/home/dani/2024-tfg-daniel-borja/datasets/Rellis_3D_os1_cloud_node_color_ply/Rellis-3D/00000/os1_cloud_node_color_ply/frame000000-1581624652_770.ply'
    
    try:
        # Crear instancia de la clase de descriptores
        descriptor_nube = LidarPointCloudDescriptors(ruta_punto_nube)
        
        # Calcular descriptores
        descriptores = descriptor_nube.calcular_descriptores()
        
        # Mostrar resultados
        print("Descriptores de la Nube de Puntos LiDAR:")
        print(f"Altura Relativa: {descriptores['altura_relativa']:.2f}")
        #print(f"Planaridad: {descriptores['planaridad']:.4f}")
        #print(f"Anisotropía: {descriptores['anisotropia']:.4f}")
        print(f"Orientación de Normales: {[f'{x:.4f}' for x in descriptores['orientacion_normal']]}")
    
    except Exception as e:
        print(f"Error al procesar la nube de puntos: {e}")

if __name__ == "__main__":
    main()
