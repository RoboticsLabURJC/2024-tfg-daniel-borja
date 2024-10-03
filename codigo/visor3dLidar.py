import numpy as np
from plyfile import PlyData
import plotly.graph_objects as go

# Leer el archivo .ply usando plyfile
plydata = PlyData.read('frame000001-1581624652_871.ply')

# Extraer las coordenadas x, y, z
vertex = plydata['vertex']
points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

# Extraer los colores RGB
colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T / 255.0

# # Obtener los valores máximos y mínimos de las coordenadas
# x_min, x_max = points[:, 0].min(), points[:, 0].max()
# y_min, y_max = points[:, 1].min(), points[:, 1].max()
# z_min, z_max = points[:, 2].min(), points[:, 2].max()

# print(f"Valores de las coordenadas:")
# print(f"X: min = {x_min}, max = {x_max}")
# print(f"Y: min = {y_min}, max = {y_max}")
# print(f"Z: min = {z_min}, max = {z_max}")

# Extraer las propiedades adicionales
intensity = vertex['intensity']
reflectivity = vertex['reflectivity']
noise = vertex['noise']
range_data = vertex['range']

# Convertir los colores a formato hexadecimal
def rgb_to_hex(r, g, b):
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return f'#{r:02x}{g:02x}{b:02x}'

colors_hex = [rgb_to_hex(r, g, b) for r, g, b in colors]

# Función que genera un gráfico 3D con Plotly
def plot_lidar_data(points, colors_hex, intensity, reflectivity, noise, range_data):
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    z_vals = points[:, 2]

    fig = go.Figure(data=[go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(
            size=0.5,
            color=colors_hex,  # Usar los colores extraídos
            opacity=0.8
        ),
        text=[f'Intensity: {i}, Reflectivity: {r}, Noise: {n}, Range: {rg}' for i, r, n, rg in zip(intensity, reflectivity, noise, range_data)]
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Eje X', range=[-10, 10]),
            yaxis=dict(title='Eje Y', range=[-10, 10]),
            zaxis=dict(title='Eje Z', range=[-10, 10]),
        ),
        title="Gráfico 3D de puntos con colores y propiedades adicionales"
    )

    # Mostrar el gráfico
    fig.show()

# Llamar a la función para generar el gráfico
plot_lidar_data(points, colors_hex, intensity, reflectivity, noise, range_data)