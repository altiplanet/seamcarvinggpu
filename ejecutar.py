import argparse
from parallelized_seam_carving_video import video_seam_carving_parallel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar un video aplicando seam carving para cambiar su resolución.")
    parser.add_argument("filename", type=str, help="Ruta del archivo de video de entrada.")
    parser.add_argument("output_file", type=str, help="Ruta del archivo de video de salida.")
    parser.add_argument("desired_width", type=int, help="Ancho deseado del video de salida.")
    parser.add_argument("desired_height", type=int, help="Altura deseada del video de salida.")

    args = parser.parse_args()

    # Llamada a la función que procesa el video
    video_seam_carving_parallel(args.filename, args.output_file, args.desired_width, args.desired_height)
