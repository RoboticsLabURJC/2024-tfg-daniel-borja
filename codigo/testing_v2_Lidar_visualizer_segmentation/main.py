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
from fileVizUpdate3_1 import vis_sequences
from carlaViz import main

# ------ global variables -------
colormaps_list = ['plasma', 'jet', 'inferno', 'viridis', 'cividis', 'turbo', 'coolwarm']
zoom_third_person = 0.012
zoom_top = 0.06

# ------- Interfaz gráfica para definir los parámetros iniciales -------
def initial_choice():
    # Configuración de la GUI inicial con dos botones
    root = tk.Tk()
    root.title("Seleccionar Modo")
    root.geometry("300x200")
    root.resizable(False, False)

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

    def on_carla_selected():
        root.destroy()
        main()

    def on_files_selected():
        root.destroy()  # Cierra la ventana inicial
        launch_interface()  # Lanza la interfaz completa para archivos

    # Botones de selección
    ttk.Button(root, text="Carla Simulator", command=on_carla_selected).pack(expand=True, pady=20)
    ttk.Button(root, text="Files", command=on_files_selected).pack(expand=True, pady=20)

    root.mainloop()

# ------ Programa Principal ------
if __name__ == "__main__":
    initial_choice()
