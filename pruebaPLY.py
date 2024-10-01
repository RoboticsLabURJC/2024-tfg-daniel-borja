import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import plotly.io as pio

# Puedes probar diferentes renderizadores según tu entorno
# pio.renderers.default = 'notebook'  # Intenta con 'inline' si estás en Jupyter

# Leer el archivo .ply usando Open3D
cloud = o3d.io.read_point_cloud("frame000001-1581624652_871.ply")

# Extraer las coordenadas x, y, z
points = np.asarray(cloud.points)

# Extraer los colores RGB
colors = np.asarray(cloud.colors)

# Normalizar los valores para que estén entre -10 y 10
# points[:, 0] = np.interp(points[:, 0], (points[:, 0].min(), points[:, 0].max()), (-10, 10))
# points[:, 1] = np.interp(points[:, 1], (points[:, 1].min(), points[:, 1].max()), (-10, 10))
# points[:, 2] = np.interp(points[:, 2], (points[:, 2].min(), points[:, 2].max()), (-10, 10))

# Convertir los colores a formato hexadecimal
def rgb_to_hex(r, g, b):
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return f'#{r:02x}{g:02x}{b:02x}'

colors_hex = [rgb_to_hex(r, g, b) for r, g, b in colors]

# Función que genera un gráfico 3D con Plotly
def plot_lidar_data(points, colors_hex):
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    z_vals = points[:, 2]

    fig = go.Figure(data=[go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(
            size=1,
            color=colors_hex,  # Usar los colores extraídos
            opacity=0.8
        ),
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='Eje X',
            yaxis_title='Eje Y',
            zaxis_title='Eje Z',
        ),
        title="Gráfico 3D de puntos con colores"
    )

    # Mostrar el gráfico
    fig.show()

# Llamar a la función para generar el gráfico
plot_lidar_data(points, colors_hex)
