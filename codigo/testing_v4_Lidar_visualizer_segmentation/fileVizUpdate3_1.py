import open3d as o3d
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys
import threading
from plyfile import PlyData
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from planar_segmentation_copy import GroundSegmentation



# ------ Parámetros y valores por defecto -------
colormaps_list = ['plasma', 'jet', 'inferno', 'viridis', 'cividis', 'turbo', 'coolwarm']
zoom_third_person = 0.012
zoom_top = 0.06

category_mapping = {
    0: [7, 22, 9, 26, 11, 21, 23, 3, 24, 31, 2, 51, 18, 5, 50],  # Drivable
    1: [43, 38, 58, 29, 41, 42, 44, 39, 55,             # No Drivable & others
        4, 45, 6, 40, 60, 61, 33, 32, 14,
        48, 47, 1, 19, 46, 10, 25,
        28, 27, 62, 52, 16, 30, 59, 17,
        13, 15, 12, 36, 57, 49, 20, 35, 37, 34, 63,
        8, 56, 0, 53, 54,]

}

category_colors = {
    0: [1.0, 0.0, 0.0],  # Rojo para Drivable
    1: [0.0, 1.0, 0.0],  # Verde para No Drivable y otros
    2: [0.0, 0.0, 1.0],  # Gris para etiquetas no mapeadas (por si acaso)
}

def map_labels_to_categories(labels, category_mapping):
    label_to_category = {}
    for category, label_list in category_mapping.items():
        for label in label_list:
            label_to_category[label] = category

    mapped_categories = np.array([label_to_category.get(lbl, -1) for lbl in labels])
    return mapped_categories

def read_label_file(file_path):
    if not file_path or not os.path.exists(file_path):
        return None
    original = np.fromfile(file_path, dtype=np.uint32)
    return map_labels_to_categories(original, category_mapping)

def get_label_colors(labels):
    return np.array([category_colors.get(int(l), [0, 0, 0]) for l in labels])

def find_label_file(bin_or_ply_path):
    base = os.path.basename(bin_or_ply_path)
    key = base.split('_vls128')[0].split('_goose')[0]
    dir_path = os.path.dirname(bin_or_ply_path)
    for f in os.listdir(dir_path):
        if f.startswith(key) and f.endswith('.label'):
            return os.path.join(dir_path, f)
    return None

# ------- Interfaz gráfica para definir los parámetros iniciales -------
def launch_interface():
    def show_controls_window():
        # Crear una nueva ventana con Tkinter en un hilo separado
        def controls_window():
            controls = tk.Tk()
            controls.title("Controles de Visualización")
            controls.geometry("400x400")
            
            # Instrucciones de control en etiquetas
            controls_list = [
                ("V", "Cambiar entre vista en tercera persona y vista superior"),
                ("C", "Cambiar el colormap"),
                ("B", "Cambiar color de fondo"),
                ("M", "Alternar entre modo automático y manual"),
                ("N", "Alternar entre muestreo 1:3 y original"),
                ("G","Alternar entre colormap y segmentación suelo"),
                ("Derecha", "Ir al siguiente fotograma (modo manual)"),
                ("Izquierda", "Ir al fotograma anterior (modo manual)"),
                ("Arriba", "Aumentar FPS"),
                ("Abajo", "Disminuir FPS"),
                ("Espacio", "Pausar/Reanudar (modo automático)"),
            ]
            
            tk.Label(controls, text="Controles:", font=("Arial", 14, "bold")).pack(pady=10)
            for key, description in controls_list:
                tk.Label(controls, text=f"{key}: {description}", font=("Arial", 10)).pack(anchor="w", padx=20)
                
            controls.mainloop()

        # Ejecuta la ventana de controles en un hilo separado
        threading.Thread(target=controls_window).start()

    def start_visualization():
        # Obtener valores de la GUI
        path = path_entry.get()
        colormap = colormap_var.get()
        fps = float(fps_var.get())
        
        # Validar el directorio
        if not os.path.isdir(path):
            messagebox.showerror("Error", "El directorio seleccionado no es válido.")
            return
        
        # Mostrar controles
        show_controls_window()
        root.destroy()
        
        # Lanzar la visualización
        vis_sequences(path, colormap, fps)

    # Configuración de la GUI con Tkinter
    root = tk.Tk()
    root.title("Configuración del Visor LiDAR")
    root.geometry("520x400")
    root.resizable(False, False)

    # Estilos personalizados
    style = ttk.Style()
    style.theme_use("clam")  # Tema moderno para ttk
    style.configure("TLabel", font=("Arial", 10), padding=5)
    style.configure("TButton", font=("Arial", 10, "bold"), padding=5)
    style.configure("TEntry", padding=5)
    style.configure("TCombobox", padding=5)

    # Contenedor principal
    frame = ttk.Frame(root, padding="20")
    frame.pack(fill="both", expand=True)

    # Campo de selección de directorio
    ttk.Label(frame, text="Selecciona el Directorio de Datos:").grid(row=0, column=0, sticky="w")
    path_entry = ttk.Entry(frame, width=40)
    path_entry.grid(row=1, column=0, padx=(0, 10), pady=5)
    ttk.Button(frame, text="Examinar", command=lambda: path_entry.insert(0, filedialog.askdirectory())).grid(row=1, column=1, pady=5)

    # Selección de Colormap
    ttk.Label(frame, text="Selecciona el Colormap:").grid(row=2, column=0, sticky="w", pady=(10, 0))
    colormap_var = tk.StringVar(value=colormaps_list[0])
    colormap_dropdown = ttk.Combobox(frame, textvariable=colormap_var, values=colormaps_list, state="readonly")
    colormap_dropdown.grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")

    # Selección de FPS con Slider
    ttk.Label(frame, text="FPS iniciales:").grid(row=4, column=0, sticky="w", pady=(10, 0))

    # Variable para el valor del slider y etiqueta para mostrar el valor actual
    fps_var = tk.IntVar(value=1)  # Inicializamos en 1
    fps_slider = ttk.Scale(frame, from_=1, to=20, orient="horizontal", variable=fps_var)  # Slider de 1 a 20
    fps_slider.grid(row=5, column=0, columnspan=1, pady=5, sticky="ew")

    # Etiqueta que muestra el valor seleccionado
    fps_value_label = ttk.Label(frame, text=f"{fps_var.get()} FPS")  # Inicializa con el valor actual
    fps_value_label.grid(row=5, column=1, padx=10, sticky="w")

    # Función para actualizar la etiqueta con el valor del slider
    def update_fps_label(*args):
        fps_value_label.config(text=f"{fps_var.get()} FPS")

    # Vinculamos el cambio de valor en el slider a la actualización de la etiqueta
    fps_var.trace("w", update_fps_label)

    # Botón para iniciar
    start_button = ttk.Button(frame, text="Iniciar Visualización", command=start_visualization)
    start_button.grid(row=6, column=0, columnspan=2, pady=(20, 0))

    root.mainloop()

# ------- Funciones de Visualización -------

def read_bin_file(file_path):
    if not file_path.endswith('.bin') or not os.path.exists(file_path):
        print(f"Error: .bin file not found at {file_path}")
        sys.exit(1)
    print(file_path)
    scan = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
    points, remissions = scan[:, 0:3], scan[:, 3]
    return points, remissions

def read_ply_file(file_path):
    if not file_path.endswith('.ply') or not os.path.exists(file_path):
        print(f"Error: .ply file not found at {file_path}")
        sys.exit(1)
    plydata = PlyData.read(file_path)
    print(file_path)
    x, y, z = plydata['vertex'].data['x'], plydata['vertex'].data['y'], plydata['vertex'].data['z']
    points = np.vstack((x, y, z)).T
    remissions = plydata['vertex'].data['intensity']
    return points, remissions

def load_path_files(path) -> list:
    def extract_sample_number(file_name, extension):
        parts = file_name.split("__" if extension == '.bin' else "-")
        num_str = ''.join(filter(str.isdigit, parts[1] if extension == '.bin' else parts[0]))
        return int(num_str) if num_str.isdigit() else 0

    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.bin') or file.endswith('.ply')]
    if files:
        ext = os.path.splitext(files[0])[1]
        return sorted(files, key=lambda file: extract_sample_number(os.path.basename(file), ext))
    return []

def set_colors(remissions, colormap_name='plasma') -> np.ndarray:
    norm_remissions = (remissions - remissions.min()) / (remissions.max() - remissions.min())
    colors = plt.get_cmap(colormap_name)(norm_remissions)[:, :3]
    return colors

def update_colors(point_cloud, remissions, colormap_name='plasma'):
    colors = set_colors(remissions, colormap_name)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

def add_new_sample(point_cloud, points, colors):
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

def configure_render_options(vis):
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1.5

def configure_camera_third_person(vis, data_type=True):
    view_control = vis.get_view_control()
    view_control.set_front([-1, 0, 0.4] if data_type else [1, 0, 0.4])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(zoom_third_person)

def configure_camera_top(vis, data_type=True):
    view_control = vis.get_view_control()
    view_control.set_up([1, 0, 0] if data_type else [-1, 0, 0])
    view_control.set_front([0, 0, 1])
    view_control.set_lookat([0, 0, 0])
    view_control.set_zoom(zoom_top)

def add_sensor(vis):
    # Añadir la posición del sensor como un cilindro rojo en el origen
    sensor_pos = o3d.geometry.TriangleMesh.create_cylinder(radius=0.2, height=0.5)  # Crear un cilindro
    sensor_pos.translate([0, 0, 0])  # Posición del sensor en el origen
    sensor_pos.paint_uniform_color([1, 0, 0])  # Color rojo para el sensor
    vis.add_geometry(sensor_pos)

def add_axis(vis):
    # Añadir ejes de coordenadas en el origen para referencia
    ejes_coordenadas = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])  # Tamaño ajustable
    vis.add_geometry(ejes_coordenadas)

def vis_sequences(path, initial_colormap='plasma', initial_fps=0.5):
    FPS = [initial_fps]
    current_colormap = [initial_colormap]
    path_file_list = load_path_files(path)
    is_bin_file = [path_file_list[0].endswith('.bin')] if path_file_list else [False]
    num_files = len(path_file_list)
    point_cloud = o3d.geometry.PointCloud()

    is_segmented = [False]
    current_colors = [None]
    label_colors_enabled = [False]
    labels_list = [None]
    all_files_processed = [False]

    label_path = find_label_file(path_file_list[0])
    labels = read_label_file(label_path)
    labels_list[0] = labels
    
    if is_bin_file[0]:
        points, remissions_data = read_bin_file(path_file_list[0])
    else:
        points, remissions_data = read_ply_file(path_file_list[0])

    remissions = [remissions_data]
    if labels is not None:
        colors = get_label_colors(labels)
    else:
        colors = set_colors(remissions[0], current_colormap[0])
    current_colors[0] = colors
    add_new_sample(point_cloud, points, colors)

    segmentador = GroundSegmentation(distancia_threshold=0.1, max_angle_deg=40.0, min_inliers=500)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='PointCloud Sequence')
    configure_render_options(vis)
    vis.add_geometry(point_cloud)
    add_sensor(vis)
    add_axis(vis)
    configure_camera_third_person(vis, is_bin_file[0])
    

    is_third_person = [True]
    colormap_index = [colormaps_list.index(current_colormap[0])]
    background = [False]
    is_paused = [False]
    is_auto_mode = [True]
    is_resampled = [False]
    frame = [0]
    last_update_time = [time.time()]

    def show_completion_message():       
        # Crear texto con la información de finalización
        completion_text = f"Procesamiento completado\n\nArchivos analizados: {num_files}"
        
        # Mostrar el texto en la consola
        print("\n" + "="*50)
        print(completion_text)
        print("="*50 + "\n")
        
        # Actualizar la visualización
        vis.update_geometry(point_cloud)
        vis.update_renderer()

    def toggle_camera(vis):
        if is_third_person[0]:
            configure_camera_top(vis, is_bin_file[0])
        else:
            configure_camera_third_person(vis, is_bin_file[0])
        is_third_person[0] = not is_third_person[0]
        vis.update_renderer()

    def toggle_colormap(vis):
        colormap_index[0] = (colormap_index[0] + 1) % len(colormaps_list)
        current_colormap[0] = colormaps_list[colormap_index[0]]
        update_colors(point_cloud, remissions[0], current_colormap[0])
        vis.update_geometry(point_cloud)
        vis.update_renderer()
        print(f'Colormap changed to: {current_colormap[0]}')

    def change_color_to_segment(vis):
        # Cambiar los colores de la nube de puntos utilizando la segmentación
        segmentador.segment_ground(point_cloud)  # Llama al método de segmentación
        vis.update_geometry(point_cloud)  # Actualiza la visualización
        print("Colores actualizados según la segmentación.")

    def toggle_background(vis):
        vis.get_render_option().background_color = [0.05, 0.05, 0.05] if background[0] else [0.95, 0.95, 0.95]
        background[0] = not background[0]
        vis.update_renderer()

    def toggle_pause(vis):
        is_paused[0] = not is_paused[0]
        print("Paused" if is_paused[0] else "Playing")

    def increase_fps(vis):
        FPS[0] += 1
        print(f"FPS increased to: {FPS[0]}")

    def decrease_fps(vis):
        FPS[0] = max(0.1, FPS[0] - 1)
        print(f"FPS decreased to: {FPS[0]}")

    def toggle_mode(vis):
        is_auto_mode[0] = not is_auto_mode[0]
        print("Mode:", "Automatic" if is_auto_mode[0] else "Manual")

    def toggle_resampling(vis):
        is_resampled[0] = not is_resampled[0]
        print("Resampling:", "1:3" if is_resampled[0] else "Original")
        update_point_cloud()

    def next_frame():
        frame[0] = (frame[0] + 1) % num_files
        update_point_cloud()

    def prev_frame():
        frame[0] = (frame[0] - 1) % num_files
        update_point_cloud()

    def toggle_segmentation(vis):
        is_segmented[0] = not is_segmented[0]
        if is_segmented[0]:
            try:
                # Aplicar segmentación
                segmentador.segment_ground(point_cloud)

                # Calcular IoU si hay etiquetas
                if labels_list[0] is not None:
                    ground_mask = segmentador.get_ground_mask(point_cloud)
                    segmentador.all_ground_mask.append(ground_mask)
                    segmentador.all_semantic_labels.append(labels_list[0])

                    segmentador.evaluate_segmentation_with_iou(point_cloud, labels_list[0])
                else:
                    print("⚠️  No hay etiquetas disponibles para calcular IoU.")

                # Actualizar visualización con colores de segmentación
                new_colors = np.asarray(point_cloud.colors)
                point_cloud.colors = o3d.utility.Vector3dVector(new_colors)

                print("Vista de segmentación activada.")
                vis.update_geometry(point_cloud)
                vis.update_renderer()

            except Exception as e:
                print(f"❌ Error en segmentación: {str(e)}")
                is_segmented[0] = False
        else:
            # Restaurar los colores originales
            point_cloud.colors = o3d.utility.Vector3dVector(current_colors[0])
            print("Vista de colormap restaurada.")
            vis.update_geometry(point_cloud)
            vis.update_renderer()


    def update_point_cloud():
        if frame[0] == num_files - 1 and not all_files_processed[0]:
            all_files_processed[0] = True
            is_paused[0] = True  # Pausar automáticamente al terminar
            show_completion_message()  # Mostrar mensaje de finalización
            segmentador.calculate_global_iou()  # Mostrar IoU global   
            return
        
        # Leer archivo
        if is_bin_file[0]:
            points, remissions_data = read_bin_file(path_file_list[frame[0]])
        else:
            points, remissions_data = read_ply_file(path_file_list[frame[0]])

        if is_resampled[0]:
            points, remissions_data = points[::3], remissions_data[::3]

        remissions[0] = remissions_data
        point_cloud.points = o3d.utility.Vector3dVector(points)

        # Cargar etiquetas siempre
        label_path = find_label_file(path_file_list[frame[0]])
        print(f"Cargando etiquetas de: {label_path}")
        labels = read_label_file(label_path)
        labels_list[0] = labels

        # Segmentación activada
        if is_segmented[0]:
            segmentador.segment_ground(point_cloud)

            # Evaluar con etiquetas actuales
            if labels_list[0] is not None:
                segmentador.evaluate_segmentation_with_iou(point_cloud, labels_list[0])
        else:
            # Mostrar con colormap si no está segmentado
            colors = set_colors(remissions[0], current_colormap[0])
            current_colors[0] = colors
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Verificar si es el último archivo
        if frame[0] == num_files - 1 and not all_files_processed[0]:
            all_files_processed[0] = True
            segmentador.calculate_global_iou()  # Llamar a la función para calcular IoU global

        # Mostrar etiquetas si están activadas
        if label_colors_enabled[0] and labels_list[0] is not None:
            point_cloud.colors = o3d.utility.Vector3dVector(get_label_colors(labels_list[0]))

        vis.update_geometry(point_cloud)
        vis.update_renderer()


    def toggle_labels(vis):
        if labels_list[0] is None:
            print("No hay etiquetas disponibles.")
            return
        label_colors_enabled[0] = not label_colors_enabled[0]
        if label_colors_enabled[0]:
            print("Vista por etiquetas activada.")
            point_cloud.colors = o3d.utility.Vector3dVector(get_label_colors(labels_list[0]))
        else:
            print("Vista por colormap restaurada.")
            point_cloud.colors = o3d.utility.Vector3dVector(current_colors[0])
        vis.update_geometry(point_cloud)
        vis.update_renderer()

    def toggle_colormap(vis):
        if not is_segmented[0]:  # Solo cambiar colormap si no estamos en modo segmentación
            colormap_index[0] = (colormap_index[0] + 1) % len(colormaps_list)
            current_colormap[0] = colormaps_list[colormap_index[0]]
            colors = set_colors(remissions[0], current_colormap[0])
            current_colors[0] = colors
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(point_cloud)
            vis.update_renderer()
            print(f'Colormap changed to: {current_colormap[0]}')

    vis.register_key_callback(ord("L"), toggle_labels)
    vis.register_key_callback(ord("G"), toggle_segmentation)
    vis.register_key_callback(ord("V"), toggle_camera)
    vis.register_key_callback(ord("C"), toggle_colormap)
    vis.register_key_callback(ord("B"), toggle_background)
    vis.register_key_callback(32, toggle_pause)
    vis.register_key_callback(265, increase_fps)
    vis.register_key_callback(264, decrease_fps)
    vis.register_key_callback(ord("M"), toggle_mode)
    vis.register_key_callback(ord("N"), toggle_resampling)
    vis.register_key_callback(262, lambda vis: next_frame() if not is_auto_mode[0] else None)
    vis.register_key_callback(263, lambda vis: prev_frame() if not is_auto_mode[0] else None)
    vis.register_key_callback(ord("I"), lambda vis: segmentador.calculate_global_iou())

    def update_frame(vis):
        if is_paused[0] or not is_auto_mode[0]:
            return
        current_time = time.time()
        if current_time - last_update_time[0] >= 1 / FPS[0]:
            next_frame()
            last_update_time[0] = current_time

    vis.register_animation_callback(update_frame)
    vis.run()
    vis.destroy_window()
    vis.destroy_window()

# ------ Programa Principal ------
#if __name__ == "__main__":
#    launch_interface()
