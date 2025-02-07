import numpy as np
from plyfile import PlyData, PlyElement

def create_sphere(radius, center, num_points=10):
    """Genera puntos en una esfera."""
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, np.pi, num_points)
    x = radius * np.sin(theta) * np.cos(phi) + center[0]
    y = radius * np.sin(theta) * np.sin(phi) + center[1]
    z = radius * np.cos(theta) + center[2]
    return np.vstack((x, y, z)).T

# Crear dos esferas
sphere1 = create_sphere(radius=1.0, center=[-5, 0, 0], num_points=100)
sphere2 = create_sphere(radius=1.0, center=[5, 0, 0], num_points=100)

# Combinar las dos esferas
points = np.vstack((sphere1, sphere2))

# Asignar intensidades (pueden ser valores aleatorios o fijos)
intensities = np.random.uniform(0, 1, len(points))

# Crear el archivo PLY
vertex = np.zeros(len(points), dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
    ('intensity', 'f4')
])
vertex['x'] = points[:, 0]
vertex['y'] = points[:, 1]
vertex['z'] = points[:, 2]
vertex['intensity'] = intensities

ply_element = PlyElement.describe(vertex, 'vertex')
PlyData([ply_element]).write('/home/dani/2024-tfg-daniel-borja/codigo/densificacion/Ejemplo/two_spheres.ply')

print("Archivo PLY generado: two_spheres.ply")