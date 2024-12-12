import torch
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from argparse_seam_carving_image import process_frame


def process_video_frame(frame, desired_width, desired_height):
    """
    Procesa un solo frame y devuelve el frame redimensionado usando seam carving en la GPU.
    """
    try:
        print("Procesando un frame individual...")
        result = process_frame(frame, desired_width, desired_height)
        print("Frame procesado correctamente.")
        return result
    except Exception as e:
        raise RuntimeError(f"Error procesando un frame: {e}")


def video_seam_carving_parallel(input_file, output_file, desired_width, desired_height):
    """
    Procesa un video utilizando seam carving para cambiar la resolución, con soporte de GPU y paralelización.
    """
    print("Abriendo video de entrada...")
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        raise ValueError(f"Error: No se pudo abrir el archivo de video '{input_file}'.")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Número total de frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para guardar el video

    # Inicializar el objeto de escritura para el video de salida
    print(f"Creando archivo de salida: {output_file}")
    out = cv2.VideoWriter(output_file, fourcc, fps, (desired_width, desired_height))

    frames = []
    print(f"Frames totales a procesar: {frame_count}")

    # Cargar frames en la memoria
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    print(f"Frames cargados en memoria: {len(frames)}")

    # Procesar frames con barra de progreso
    errors_encountered = False
    with ThreadPoolExecutor() as executor, tqdm(total=len(frames), desc="Procesando frames") as pbar:
        futures = {
            executor.submit(process_video_frame, frame, desired_width, desired_height): idx
            for idx, frame in enumerate(frames)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                resized_frame = future.result()
                out.write(resized_frame)
                pbar.update(1)
            except Exception as e:
                errors_encountered = True
                print(f"Error procesando el frame {idx + 1}: {e}")
                break  # Detener el procesamiento si ocurre un error

    cap.release()
    out.release()

    if errors_encountered:
        print(f"Procesamiento detenido debido a errores. Video incompleto en '{output_file}'.")
    else:
        print(f"Video procesado con éxito y guardado en: {output_file}")


def video_seam_carving_sequential(input_file, output_file, desired_width, desired_height):
    """
    Procesa un video frame por frame (sin paralelismo) para simplificar la depuración.
    """
    print("Abriendo video de entrada...")
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        raise ValueError(f"Error: No se pudo abrir el archivo de video '{input_file}'.")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Número total de frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para guardar el video

    print(f"Creando archivo de salida: {output_file}")
    out = cv2.VideoWriter(output_file, fourcc, fps, (desired_width, desired_height))

    print(f"Frames totales a procesar: {frame_count}")
    with tqdm(total=frame_count, desc="Procesando frames (secuencialmente)") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                print("Procesando frame secuencialmente...")
                resized_frame = process_frame(frame, desired_width, desired_height)
                out.write(resized_frame)
                pbar.update(1)
            except Exception as e:
                print(f"Error procesando un frame: {e}")
                break

    cap.release()
    out.release()
    print(f"Video procesado y guardado en: {output_file}")
