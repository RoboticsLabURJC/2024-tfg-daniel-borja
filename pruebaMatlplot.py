import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import numpy as np

# Coordenadas de los puntos que quieres dibujar
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])
z = np.array([1, 4, 9, 16, 25])

# Colores de los puntos
colors = np.array(['r', 'g', 'b', 'y', 'm'])  # r: rojo, g: verde, b: azul, y: amarillo, m: magenta

# Crear la figura y el eje 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c=colors)

# Añadir un slider para manejar la escala
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Escala', 0.1, 2.0, valinit=1.0)

def update(val):
    scale = slider.val
    ax.clear()
    ax.scatter(x, y, z, c=colors)
    ax.set_xlim([min(x) * scale, max(x) * scale])
    ax.set_ylim([min(y) * scale, max(y) * scale])
    ax.set_zlim([min(z) * scale, max(z) * scale])
    plt.draw()

slider.on_changed(update)

plt.show()
