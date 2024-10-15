import carla
import math
import random
import time
import numpy as np
import cv2
import pygame
import signal

# Inicializar pygame
pygame.init()
display = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Visor LIDAR en Tiempo Real")

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

vehicle_bp = bp_lib.filter('vehicle.*')[38]
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

spectator = world.get_spectator()
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
lidar_bp.set_attribute('range', '100')  # Ajusta el rango según tus necesidades
lidar_bp.set_attribute('rotation_frequency', '40')  # Ajusta la frecuencia de rotación según tus necesidades
lidar_bp.set_attribute('channels', '32')  # Ajusta el número de canales según el sensor
lidar_bp.set_attribute('points_per_second', '1000000')  # Aumenta los puntos por segundo para mayor densidad
lidar_bp.set_attribute('upper_fov', '10')  # Campo de visión superior en grados
lidar_bp.set_attribute('lower_fov', '-30')  # Campo de visión inferior en grados
lidar_bp.set_attribute('horizontal_fov', '360')  # Asegúrate de que el FOV horizontal esté en 360 grados

lidar_init_trans = carla.Transform(carla.Location(x=0, z=2.5))  # Ajusta la posición del LIDAR según sea necesario
lidar = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=vehicle)

# Carpeta para guardar los datos del LIDAR
carpeta_lidar = '/home/daniel/2024-tfg-daniel-borja/codigo/lidar_datos'

# Lista para almacenar los datos del LIDAR
lidar_data = []

def guardar_lidar(data):
    global ultimo_guardado
    points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
    lidar_data.append(points)
    if time.time() - ultimo_guardado >= 1:
        frame = data.frame
        data.save_to_disk(f'{carpeta_lidar}/lidar_%06d.ply' % frame)
        print(f'Guardando datos LIDAR del frame {frame}')
        ultimo_guardado = time.time()

lidar.listen(lambda data: guardar_lidar(data))
print("Listener del LIDAR configurado")

# Activa el modo autopilot del vehículo
vehicle.set_autopilot(True)

def transformar_puntos(points, vehicle_transform):
    # Obtener ángulo de rotación del vehículo en radianes
    yaw = math.radians(vehicle_transform.rotation.yaw)
    # Crear matriz de rotación del vehículo
    rotation_matrix = np.array([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw), math.cos(yaw)]
    ])
    # Aplicar rotación del vehículo a los puntos
    transformed_points = np.dot(points[:, :2], rotation_matrix)
    
    # Crear matriz de rotación de 90 grados a la izquierda
    rotation_90_left = np.array([
        [0, -1],
        [1, 0]
    ])
    # Aplicar rotación de 90 grados a la izquierda
    transformed_points = np.dot(transformed_points, rotation_90_left)
    
    return transformed_points

def distancia_a_color(distancia, max_distancia):
    intensidad = max(0, min(255, int(255 * (1 - distancia / max_distancia))))
    return (intensidad, 0, 255 - intensidad)

def cerrar_cliente(signal_received, frame):
    print("Cerrando cliente...")
    camera.stop()
    lidar.stop()
    vehicle.destroy()
    print("Vehículo eliminado")
    pygame.quit()
    exit(0)

# Configurar la señal para cerrar el cliente
signal.signal(signal.SIGINT, cerrar_cliente)

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cerrar_cliente(None, None)

        display.fill((255, 255, 255))

        if lidar_data:
            points = lidar_data[-1]  # Usar los datos LIDAR más recientes
            # Obtener la transformación del vehículo
            vehicle_transform = vehicle.get_transform()
            # Transformar los puntos para alinear con la dirección del vehículo y rotar 90 grados a la izquierda
            transformed_points = transformar_puntos(points, vehicle_transform)

            # Proyectar los puntos en el plano 2D desde un punto de vista lateral con inclinación
            max_distancia = np.sqrt(np.max(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2))
            for point in transformed_points:
                distancia = np.linalg.norm(point[:2])
                color = distancia_a_color(distancia, max_distancia)
                x = int(400 + (point[0] * 10))
                y = int(300 - (points[np.where(transformed_points == point)[0][0], 2] * 10) + (point[1] * 10 * 0.5))  # Ajustar para inclinar la vista
                if 0 <= x < 800 and 0 <= y < 600:  # Asegúrate de que los puntos estén dentro del rango visible
                    pygame.draw.circle(display, color, (x, y), 1)

        pygame.display.flip()

        # Actualizar la posición del espectador con la del vehículo en cada ciclo del bucle sin esperar
        vehicle_transform = vehicle.get_transform()
        spectator_transform = carla.Transform(vehicle_transform.location + carla.Location(x=-5, z=2.5), vehicle_transform.rotation)
        spectator.set_transform(spectator_transform)

        world.tick()

except KeyboardInterrupt:
    cerrar_cliente(None, None)
