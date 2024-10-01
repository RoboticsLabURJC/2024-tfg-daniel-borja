import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Puedes probar diferentes renderizadores según tu entorno
# pio.renderers.default = 'notebook'  # Intenta con 'inline' si estás en Jupyter

# Función para convertir RGB a formato hexadecimal
def rgb_to_hex(r, g, b):
    # Asegurarse de que los valores r, g, b sean enteros
    r, g, b = int(r), int(g), int(b)
    return f'#{r:02x}{g:02x}{b:02x}'

# Función que genera un gráfico 3D con Plotly
def plot_lidar_data(x_vals, y_vals, z_vals, colors):
    fig = go.Figure(data=[go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
        )
    )])

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    fig.show()

# Ejemplo de datos personalizados
x_vals = [10, 20, 30, 40, 50]
y_vals = [10, 25, 35, 45, 55]
z_vals = [10, 30, 40, 50, 60]
colors = [rgb_to_hex(255, 0, 0), rgb_to_hex(0, 255, 0), rgb_to_hex(0, 0, 255), rgb_to_hex(255, 255, 0), rgb_to_hex(0, 255, 255)]

# Llamada a la función para generar el gráfico
plot_lidar_data(x_vals, y_vals, z_vals, colors)
