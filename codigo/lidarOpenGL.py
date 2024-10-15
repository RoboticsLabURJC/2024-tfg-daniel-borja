import carla
import math
import random
import time
import numpy as np
import pygame
import signal
import threading
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Inicializar pygame con contexto OpenGL
pygame.init()
display = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Visor LIDAR en Tiempo Real con OpenGL")
clock = pygame.time.Clock()  # Reloj para limitar FPS

# Configurar el color de fondo blanco en OpenGL
glClearColor(1.0, 1.0, 1.0, 1.0)  # Fondo blanco

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

vehicle_bp = bp_lib.filter('vehicle.*')[38]
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

camera_bp = bp_lib.find('sensor.camera.rgb')
camera_init_trans = carla.Transform(carla.Location(x=-0.5, z=2.5), carla.Rotation(pitch=-10))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

carpeta_guardado = '/home/daniel/2024-tfg-daniel-borja/codigo/capturas'

# Variable global para controlar el tiempo de guardado
ultimo_guardado = time.time()

def guardar_imagen(image):
    global ultimo_guardado
    if time.time() - ultimo_guardado >= 1:
        image.save_to_disk(f'{carpeta_guardado}/%06d.png' % image.frame)
        print(f'Guardando imagen {image.frame}')
        ultimo_guardado = time.time()

camera.listen(lambda image: guardar_imagen(image))
print("Listener de la cámara configurado")

# Añadimos el sensor LIDAR
lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '100')  # Rango del LIDAR
lidar_bp.set_attribute('rotation_frequency', '60')  # Aumentar frecuencia de rotación
lidar_bp.set_attribute('channels', '64')  # Número de canales
lidar_bp.set_attribute('points_per_second', '2000000')  # Puntos por segundo
lidar_bp.set_attribute('upper_fov', '10')
lidar_bp.set_attribute('lower_fov', '-30')
lidar_bp.set_attribute('horizontal_fov', '360')

lidar_init_trans = carla.Transform(carla.Location(x=0, z=2.5))
lidar = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=vehicle)

# Carpeta para guardar los datos del LIDAR
carpeta_lidar = '/home/daniel/2024-tfg-daniel-borja/codigo/lidar_datos'

# Lista para almacenar los datos del LIDAR
lidar_data = []
lidar_data_lock = threading.Lock()

def guardar_lidar(data):
    global ultimo_guardado
    points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
    with lidar_data_lock:
        lidar_data.append(points)  # Protege la lista compartida usando un candado
    if time.time() - ultimo_guardado >= 1:
        frame = data.frame
        data.save_to_disk(f'{carpeta_lidar}/lidar_%06d.ply' % frame)
        print(f'Guardando datos LIDAR del frame {frame}')
        ultimo_guardado = time.time()

# Listener del LIDAR en un hilo separado para no bloquear la visualización
lidar_thread = threading.Thread(target=lambda: lidar.listen(guardar_lidar))
lidar_thread.start()
print("Listener del LIDAR configurado")

# Activa el modo autopilot del vehículo
vehicle.set_autopilot(True)

def transformar_puntos(points, vehicle_transform):
    yaw = math.radians(vehicle_transform.rotation.yaw)
    rotation_matrix = np.array([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw), math.cos(yaw)]
    ])
    transformed_points = np.dot(points[:, :2], rotation_matrix)
    
    rotation_90_left = np.array([
        [0, -1],
        [1, 0]
    ])
    transformed_points = np.dot(transformed_points, rotation_90_left)
    
    return transformed_points

def distancia_a_color(distancia, max_distancia):
    intensidad = max(0, min(1, distancia / max_distancia))
    r = int(255 * (intensidad))
    g = 0
    b = int(255 * (1-intensidad))
    return (r, g, b)

def cerrar_cliente(signal_received, frame):
    print("Cerrando cliente...")
    camera.stop()
    lidar.stop()
    vehicle.destroy()
    print("Vehículo eliminado")
    pygame.quit()
    exit(0)

signal.signal(signal.SIGINT, cerrar_cliente)

def draw_lidar_points(points, vehicle_transform):
    max_distancia = np.sqrt(np.max(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2))
    transformed_points = transformar_puntos(points, vehicle_transform)

    # Aumentar el tamaño de los puntos
    glPointSize(2.0)
    
    glBegin(GL_POINTS)
    for i, point in enumerate(transformed_points):
        distancia = np.linalg.norm(point[:2])
        color = distancia_a_color(distancia, max_distancia)
        glColor3f(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        
        x = point[0] * 5  # Reducir la escala para que los puntos estén más juntos
        y = point[1] * 5 * 0.5
        z = points[i, 2] * 5
        
        glVertex3f(x, y, z)
    glEnd()

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cerrar_cliente(None, None)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Posicionar la cámara para una vista cenital
        gluPerspective(90, (800 / 600), 0.1, 500.0)  # Ángulo de visión amplio para ver mejor
        glTranslatef(0, 0, 20)  # Elevar la cámara en el eje Z
        glRotatef(90, 1, 0, 0)  # Apuntar hacia abajo

        if lidar_data:
            with lidar_data_lock:
                points = lidar_data[-1]

            # Reducir la cantidad de puntos para suavizar el renderizado
            downsample_factor = 3  # Aumentar la densidad al reducir el downsample factor
            points = points[::downsample_factor]

            # Obtener la transformación del vehículo
            vehicle_transform = vehicle.get_transform()

            # Dibujar los puntos LIDAR
            draw_lidar_points(points, vehicle_transform)

        pygame.display.flip()
        clock.tick(30)  # Limitar a 30 FPS
        world.tick()

except KeyboardInterrupt:
    cerrar_cliente(None, None)
