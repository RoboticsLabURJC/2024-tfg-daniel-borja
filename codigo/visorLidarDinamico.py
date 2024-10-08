import os
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from flask import Flask, render_template, redirect, url_for

# Inicializa la aplicación Flask
app = Flask(__name__)

# Clase base para la lectura de datos LiDAR del dataset GOOSE o RELLIS
class LidarDataReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.read_lidar_data()

    def read_lidar_data(self):
        if self.file_path.endswith('.bin'):
            return self.read_lidar_data_from_bin()
        elif self.file_path.endswith('.ply'):
            return self.read_lidar_data_from_ply()
        else:
            print("Formato de archivo no soportado.")
            return np.array([])

    def read_lidar_data_from_bin(self):
        try:
            scan = np.fromfile(self.file_path, dtype=np.float32)
            points = scan.reshape((-1, 4))
            return points
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {self.file_path}")
            return np.array([])

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
        super().__init__(file_path)

    def intensity_to_color(self, intensity):
        intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        colormap = plt.get_cmap('plasma')
        rgba_colors = colormap(intensity_normalized)
        hex_colors = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b, _ in rgba_colors]
        return hex_colors

    def plot_lidar_data(self):
        if self.data.size == 0:
            print("No hay datos LiDAR para visualizar.")
            return None
        
        x_vals = self.data[:, 0]
        y_vals = self.data[:, 1]
        z_vals = self.data[:, 2]

        if self.data.shape[1] == 4:
            intensities = self.data[:, 3]
            colors = self.intensity_to_color(intensities)
            text_vals = [f'Intensidad: {intensity}' for intensity in intensities]
        else:
            colors = 'blue'
            text_vals = ['Sin intensidad'] * len(x_vals)

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
            text=text_vals,
        )])

        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Eje X', range=[-10, 10]),
                yaxis=dict(title='Eje Y', range=[-10, 10]),
                zaxis=dict(title='Eje Z', range=[-10, 10]),
            ),
            title="Visualización 3D de datos LiDAR",
            width=1200,  # Aumentar el ancho del gráfico
            height=800,  # Aumentar la altura del gráfico
        )

        return fig.to_html(full_html=False)

# Función para listar los archivos LiDAR en una carpeta
def get_lidar_files_from_folder(folder_path):
    # Filtrar solo archivos con extensión .bin y .ply
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.bin') or f.endswith('.ply')]

# Utilizar la carpeta actual donde se ejecuta el script
lidar_folder = os.getcwd()  # Esto obtendrá la ruta del directorio actual
lidar_files = get_lidar_files_from_folder(lidar_folder)

# Verifica si se encontraron archivos
if not lidar_files:
    raise FileNotFoundError(f"No se encontraron archivos LiDAR (.bin o .ply) en la carpeta {lidar_folder}")


@app.route('/')
def index():
    return redirect(url_for('show_lidar', file_index=0))

@app.route('/lidar/<int:file_index>')
def show_lidar(file_index):
    if file_index < 0 or file_index >= len(lidar_files):
        return "Índice de archivo fuera de rango", 404
    
    file_path = lidar_files[file_index]
    file_name = file_path.split('/')[-1]  # Obtener solo el nombre del archivo, no la ruta completa

    visualizer = LidarVisualizer(file_path)
    plot_html = visualizer.plot_lidar_data()

    next_index = file_index + 1 if file_index + 1 < len(lidar_files) else 0
    prev_index = file_index - 1 if file_index > 0 else len(lidar_files) - 1

    return render_template('lidar.html', plot_html=plot_html, next_index=next_index, prev_index=prev_index, file_name=file_name)

if __name__ == '__main__':
    app.run(debug=True)
