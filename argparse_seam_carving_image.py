# Función de procesamiento de imagen (seam carving)
import numpy as np
import torch


def calculate_energy_gpu(I):
    """
    Calcula la energía de la imagen usando GPU.
    """
    # Convertir a tensor de PyTorch si I es un numpy.ndarray
    if isinstance(I, np.ndarray):
        I = torch.tensor(I).float().cuda()  # Convertir y mover a GPU

    energy = torch.zeros_like(I[:, :, 0])  # Usar solo un canal para la energía

    # Convertir la imagen a escala de grises
    gray = I.mean(dim=2, keepdim=True)

    # Calcular el gradiente utilizando diferencias finitas
    for y in range(1, gray.shape[0] - 1):
        for x in range(1, gray.shape[1] - 1):
            val = 0
            val += 2 * abs(gray[y, x + 1] - gray[y, x])
            val += 2 * abs(gray[y, x - 1] - gray[y, x])
            val += abs(gray[y + 1, x] - gray[y, x])
            val += abs(gray[y - 1, x] - gray[y, x])
            energy[y, x] = val

    return energy


def find_vertical_seam(energy):
    height, width = energy.shape
    cost = np.zeros_like(energy)
    cost[0, :] = energy[0, :]

    # Fill cumulative cost matrix
    for i in range(1, height):
        for j in range(1, width - 1):
            cost[i, j] = energy[i, j] + min(cost[i - 1, j - 1], cost[i - 1, j], cost[i - 1, j + 1])

    # Initialize seam path
    seam = np.zeros(height, dtype=int)
    seam[-1] = np.argmin(cost[-1, :])

    # Backtrack to find the seam
    for i in range(height - 2, -1, -1):
        j = seam[i + 1]
        seam[i] = j + np.argmin(cost[i, j - 1:j + 2]) - 1

    # Validation: make sure the seam isn't empty
    if len(seam) == 0:
        raise ValueError("Vertical seam is empty, possibly due to errors in energy calculation.")

    return seam

def add_horizontal(I, target_height):
    energy = calculate_energy_gpu(I)
    seam = find_vertical_seam(energy)
    # Insert new seam into image, increasing height
    return I

def remove_horizontal_img(I, target_height):
    energy = calculate_energy_gpu(I)
    seam = find_vertical_seam(energy)
    # Remove seam from image, decreasing height
    return I

def add_vertical(I, target_width):
    energy = calculate_energy_gpu(I)
    seam = find_vertical_seam(energy)
    # Insert new seam into image, increasing width
    return I

def remove_vertical_img(I, target_width):
    energy = calculate_energy_gpu(I)
    seam = find_vertical_seam(energy)
    # Remove seam from image, decreasing width
    return I

def process_frame(frame, target_width, target_height):
    """
    Procesa un solo fotograma de video aplicando seam carving.
    """
    from argparse_seam_carving_image import (
        remove_horizontal_img,
        remove_vertical_img,
        add_horizontal,
        add_vertical,
    )

    frame = np.array(frame)  # Convertir el frame a un array numpy si no lo es
    current_height, current_width, _ = frame.shape
    
    # Reducción o expansión de la imagen en función de las dimensiones objetivo
    if current_height > target_height:
        frame = remove_horizontal_img(frame, target_height)
    if current_height < target_height:
        frame = add_horizontal(frame, target_height)
    if current_width > target_width:
        frame = remove_vertical_img(frame, target_width)
    if current_width < target_width:
        frame = add_vertical(frame, target_width)
    
    return frame


def convert_to_h264(input_path, output_path):
    try:
        # Open devnull to suppress output
        with open(os.devnull, "w") as devnull:
            # Run ffmpeg with output and error streams redirected to devnull
            command = [
                "ffmpeg",
                "-i",
                input_path,
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                "-preset",
                "medium",
                "-hide_banner",
                "-loglevel",
                "error",
                output_path,
            ]
            subprocess.run(command, stdout=devnull, stderr=devnull, check=True)

        return output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during conversion: {e.stderr}")
        st.error("Failed to convert video to H.264 format.")
        return None





