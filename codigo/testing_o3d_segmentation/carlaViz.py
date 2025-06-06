import carla
import numpy as np
import open3d as o3d
import time
import random
from datetime import datetime
from matplotlib import colormaps as cm
import sys
import signal
import pygame

VIRIDIS = np.array(cm.get_cmap('inferno').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

# Variables globales
actor_list = []
manual_mode = False

def lidar_callback(lidar_data, point_cloud):
    data = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Reflejar los datos en el eje X
    data[:, 0] = -data[:, 0]

    intensity = data[:, -1]
    int_color = np.c_[
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 2])]

    point_cloud.points = o3d.utility.Vector3dVector(data[:, :-1])
    point_cloud.colors = o3d.utility.Vector3dVector(int_color)

def spawn_vehicle_lidar_camera(world, bp, traffic_manager, delta):
    vehicle_bp = bp.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    lidar_bp = bp.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('rotation_frequency', str(1 / delta))
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '500000')
    lidar_position = carla.Transform(carla.Location(x=-0.5, z=1.8))
    lidar = world.spawn_actor(lidar_bp, lidar_position, attach_to=vehicle)

    camera_bp = bp.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=-4.0, z=2.5))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    vehicle.set_autopilot(True, traffic_manager.get_port())
    return vehicle, lidar, camera

def set_camera_view(viz, third_person):
    ctr = viz.get_view_control()
    if third_person:
        # Configuración de cámara en tercera persona
        ctr.set_zoom(0.06)
        ctr.set_front([1.0, 0.0, 0.3])
        ctr.set_lookat([0, 0, 0])  
        ctr.set_up([0, 0, 1])
    else:
        # Configuración de cámara en primera persona (ajusta según necesidad)
        ctr.set_zoom(0.3)
        ctr.set_front([0, 0, 1])
        ctr.set_lookat([0, 0, 0])  
        ctr.set_up([-1, 0, 0])

def camera_callback(image, display_surface):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Quitar el canal alfa
    array = array[:, :, ::-1]  # Convertir de BGRA a RGB
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display_surface.blit(surface, (0, 0))
    # Actualizar solo esta superficie en vez de toda la pantalla
    pygame.display.update(display_surface.get_rect())

def vehicle_control(vehicle):
    global manual_mode
    control = carla.VehicleControl()  # Crear un control en blanco

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cleanup()
            sys.exit()

        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            manual_mode = not manual_mode
            vehicle.set_autopilot(not manual_mode)
            mode = "manual" if manual_mode else "automático"
            print(f"\nCoche en modo {mode}.")
            time.sleep(0.3)  # Evitar múltiples activaciones por pulsación rápida

    # Aplicar controles si estamos en modo manual
    keys = pygame.key.get_pressed()
    if manual_mode:
        control.throttle = 1.0 if keys[pygame.K_w] else 0.0
        control.brake = 1.0 if keys[pygame.K_s] else 0.0
        control.steer = -0.3 if keys[pygame.K_a] else 0.3 if keys[pygame.K_d] else 0.0
        vehicle.apply_control(control)

def main():
    pygame.init()
    # Configurar Pygame sin usar OpenGL explícitamente
    pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 0)
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height), pygame.SRCALPHA)
    pygame.display.set_caption("CARLA Vehículo Control")

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    settings = world.get_settings()
    delta = 0.05
    settings.fixed_delta_seconds = delta
    settings.synchronous_mode = True
    world.apply_settings(settings)

    global actor_list, third_person_view
    vehicle, lidar, camera = spawn_vehicle_lidar_camera(world, blueprint_library, traffic_manager, delta)
    actor_list.append(vehicle)
    actor_list.append(lidar)
    actor_list.append(camera)

    camera.listen(lambda image: camera_callback(image, screen))

    point_cloud = o3d.geometry.PointCloud()
    lidar.listen(lambda data: lidar_callback(data, point_cloud))

    # Utilizar VisualizerWithKeyCallback
    viz = o3d.visualization.VisualizerWithKeyCallback()
    viz.create_window(window_name='Lidar simulado en Carla', width=960, height=540, left=480, top=270)
    viz.get_render_option().background_color = [0.05, 0.05, 0.05]
    viz.get_render_option().point_size = 1.35
    viz.get_render_option().show_coordinate_frame = True

    third_person_view = True

    # Añadir el point cloud al visualizador en la primera iteración
    lidar_data_received = False

    # Configurar el callback de tecla 'v'
    def toggle_camera_view(_):
        global third_person_view
        third_person_view = not third_person_view
        set_camera_view(viz, third_person_view)
        print("Cambiando a tercera persona" if third_person_view else "Cambiando a primera persona")
        return True  # Devolver True para continuar el evento de renderizado

    # Asignar el callback al visualizador
    viz.register_key_callback(ord("V"), toggle_camera_view)

    dt0 = datetime.now()
    frame = 0

    while True:
        vehicle_control(vehicle)

        if frame == 5 and not lidar_data_received:
            viz.add_geometry(point_cloud)
            lidar_data_received = True
            print("Geometry added to the visualizer")
            set_camera_view(viz, third_person_view)

        viz.update_geometry(point_cloud)
        viz.poll_events()
        viz.update_renderer()
        time.sleep(0.03)
        world.tick()

        process_time = datetime.now() - dt0
        if process_time.total_seconds() > 0:
            fps = 1.0 / process_time.total_seconds()
            sys.stdout.write(f'\rFPS: {fps:.2f} ')
            sys.stdout.flush()
        dt0 = datetime.now()
        frame += 1

        if not viz.poll_events():
            print("Exiting visualization")
            break

    cleanup()

def cleanup():
    global actor_list
    print("\nLimpiando actores...")
    for actor in actor_list:
        if actor is not None:
            actor.destroy()
    actor_list = []
    print("Actores eliminados.")

def signal_handler(sig, frame):
    print("\nInterrupción recibida. Finalizando...")
    cleanup()
    sys.exit(0)

#if __name__ == "__main__":
#    signal.signal(signal.SIGINT, signal_handler)
#    main()
