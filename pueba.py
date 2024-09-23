import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Puedes probar diferentes renderizadores según tu entorno
# pio.renderers.default = 'notebook'  # Intenta con 'inline' si estás en Jupyter

num_points = 500
points_range = 100

# Función que crea puntos LiDAR simulados
def create_point(): 
    x = np.random.rand() * points_range - points_range/2
    y = np.random.rand() * points_range - points_range/2
    z = np.random.rand() * points_range - points_range/2
    r = int(round(np.random.rand() * 255))
    g = int(round(np.random.rand() * 255))
    b = int(round(np.random.rand() * 255))
    point = np.array([x, y, z, r, g, b])
    return point

# Función para convertir RGB a formato hexadecimal
def rgb_to_hex(r, g, b):
    # Asegurarse de que los valores r, g, b sean enteros
    r, g, b = int(r), int(g), int(b)
    return f'#{r:02x}{g:02x}{b:02x}'

# Función que crea una nube de puntos aleatorios
def create_array(n):
    points = []
    for idx in range(n):
        point = create_point()
        points.append(point)
    lidar_array = np.array(points)
    return lidar_array

data = create_array(num_points)

# Función que genera un gráfico 3D con Plotly
def plot_lidar_data(data):
    x_vals = data[:, 0]
    y_vals = data[:, 1]
    z_vals = data[:, 2]
    r_vals = data[:, 3]
    g_vals = data[:, 4]
    b_vals = data[:, 5]

    colors = [rgb_to_hex(r, g, b) for r, g, b in zip(r_vals, g_vals, b_vals)]

    # Crear las etiquetas de texto para mostrar los valores RGB
    text_vals = [f'RGB: ({r}, {g}, {b})' for r, g, b in zip(r_vals, g_vals, b_vals)]

    fig = go.Figure(data=[go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            opacity=0.8
        ),
        text=text_vals,  # Mostrar las componentes RGB en los tooltips
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Eje X',
            yaxis_title='Eje Y',
            zaxis_title='Eje Z',
        ),
        title="Gráfico 3D de puntos"
    )
    
    # Mostrar el gráfico
    fig.show()

# Llamar a la función para generar el gráfico
plot_lidar_data(data)