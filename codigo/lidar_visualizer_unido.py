import numpy as np
from plyfile import PlyData
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import argparse


# Clase base para la lectura de datos LiDAR del dataset GOOSE
class LidarDataReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.read_lidar_data()


# Función para leer datos LiDAR desde un archivo .bin o .ply
    def read_lidar_data(self):
        if self.file_path.endswith('.bin'):
            return self.read_lidar_data_from_bin()
        elif self.file_path.endswith('.ply'):
            return self.read_lidar_data_from_ply()
        else:
            print("Formato de archivo no soportado.")
            return np.array([])



# Función para leer datos LiDAR desde un archivo .bin del dataset GOOSE
    def read_lidar_data_from_bin(self):
        try:
            # Leer los datos del archivo binario
            scan = np.fromfile(self.file_path, dtype=np.float32)
            # Los datos LiDAR suelen tener 4 columnas: X, Y, Z, Intensidad
            points = scan.reshape((-1, 4))
            return points
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {self.file_path}")
            return np.array([])
        
        
# Función para leer datos LiDAR desde un archivo .ply
    def read_lidar_data_from_ply(self):
        try:
            plydata = PlyData.read(self.file_path)
            vertex = plydata['vertex']
            x = np.array(vertex['x'], dtype=np.float32)
            y = np.array(vertex['y'], dtype=np.float32)
            z = np.array(vertex['z'], dtype=np.float32)

            if 'intensity' in vertex.data.dtype.names:
                intensity = np.array(vertex['intensity'], dtype=np.float32)
                points = np.vstack((x, y, z, intensity)).T
            else:
                points = np.vstack((x, y, z)).T
            return points
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {self.file_path}")
            return np.array([])
        





# Clase derivada para la visualización de datos LiDAR
class LidarVisualizer(LidarDataReader):
    def __init__(self, file_path):
        # Inicializar la clase base con el archivo de datos LiDAR
        super().__init__(file_path)

    # Función que convierte intensidad a color en formato hexadecimal
    def intensity_to_color(self, intensity):
        # Normalizamos la intensidad
        intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        # Usamos una colormap de matplotlib (ahora correcto)
        colormap = plt.get_cmap('plasma')
        rgba_colors = colormap(intensity_normalized)
        # Convertimos los valores rgba a hex
        hex_colors = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b, _ in rgba_colors]
        return hex_colors

    # Función que genera un gráfico 3D con Plotly
    def plot_lidar_data(self):
        if self.data.size == 0:
            print("No hay datos LiDAR para visualizar.")
            return
        
        # Llamar al método para imprimir valores mínimos y máximos
        self.print_min_max_coordinates()

        # Separamos los puntos XYZ e intensidades
        x_vals = self.data[:, 0]
        y_vals = self.data[:, 1]
        z_vals = self.data[:, 2]

        # Normalizar las coordenadas a [-1, 1]
        # min_x, max_x = x_vals.min(), x_vals.max()
        # min_y, max_y = y_vals.min(), y_vals.max()
        # min_z, max_z = z_vals.min(), z_vals.max()

        # x_norm = 2 * (x_vals - min_x) / (max_x - min_x) - 1
        # y_norm = 2 * (y_vals - min_y) / (max_y - min_y) - 1
        # z_norm = 2 * (z_vals - min_z) / (max_z - min_z) - 1

        if self.data.shape[1] == 4:
            intensities = self.data[:, 3]
            colors = self.intensity_to_color(intensities)
            text_vals = [f'Intensidad: {intensity}' for intensity in intensities]
        else:
            colors = 'blue'  # Color por defecto si no hay intensidades
            text_vals = ['Sin intensidad'] * len(x_vals)

        # Crear la visualización 3D con Plotly
        fig = go.Figure(data=[go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='markers',
            marker=dict(
                size=0.8,
                color=colors,
                opacity=0.8
            ),
            text=text_vals,  # Mostrar las intensidades en los tooltips
        )])

        

        # Configuración del diseño
        fig.update_layout(
            scene=dict(
            xaxis=dict(title='Eje X', range=[-10, 10]),
            yaxis=dict(title='Eje Y', range=[-10, 10]),
            zaxis=dict(title='Eje Z', range=[-10, 10]),
            ),
            title="Gráfico 3D de puntos LiDAR del dataset GOOSE o RELLIS"
        )

        # Personalizar ejes
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    showline=True,
                    zeroline=True,
                    showgrid=False,
                    zerolinecolor='black',
                    zerolinewidth=2
                ),
                yaxis=dict(
                    showline=True,
                    zeroline=True,
                    showgrid=False,
                    zerolinecolor='black',
                    zerolinewidth=2
                ),
                zaxis=dict(
                    showline=True,
                    zeroline=True,
                    showgrid=False,
                    zerolinecolor='black',
                    zerolinewidth=2
                )
            )
        )

        # Mostrar el gráfico
        fig.show()


    # Método para imprimir los valores máximos y mínimos de las coordenadas
    def print_min_max_coordinates(self):
        if self.data.size == 0:
            print("No hay datos LiDAR para analizar.")
            return

        # Obtener los valores mínimos y máximos de las coordenadas X, Y, Z
        min_x, max_x = self.data[:, 0].min(), self.data[:, 0].max()
        min_y, max_y = self.data[:, 1].min(), self.data[:, 1].max()
        min_z, max_z = self.data[:, 2].min(), self.data[:, 2].max()

        # Imprimir los resultados
        print(f"Valores de coordenadas:")
        print(f"X: mínimo = {min_x}, máximo = {max_x}")
        print(f"Y: mínimo = {min_y}, máximo = {max_y}")
        print(f"Z: mínimo = {min_z}, máximo = {max_z}")


if __name__ == "__main__":
    # Crear el parser de argumentos
    parser = argparse.ArgumentParser(description="Visualización de datos LiDAR")
    parser.add_argument("file_path", type=str, help="Ruta del archivo LiDAR (.bin o .ply)")

    # Parsear los argumentos
    args = parser.parse_args()

    # Crear una instancia de la clase LidarVisualizer y generar la visualización
    visualizer = LidarVisualizer(args.file_path)
    visualizer.plot_lidar_data()
