from utils.graphics_utils import plot_times, plot_entropy_loss, plot_points, plot_times_for_n_bits, \
    plot_times_for_p_error


def main():
    # Gráfica de pérdida de entrenamiento (entropy loss)
    directory_path = "./results/losses/image_size"
    plot_entropy_loss(directory_path)

    # Tiempo de inferencia vs tamaño de imagen
    directory_path = "./results/times/image_size"
    plot_times(directory_path)

    # N_bits vs precision
    directory_path = "./results/times/n_bits"
    plot_points(directory_path, "N_bits vs precision", "n_bits", "precision")

    # N_bits vs tiempo de inferencia
    plot_times_for_n_bits(directory_path)

    # P_error vs precision
    directory_path = "./results/times/p_error"
    plot_points(directory_path, "P_error vs precision", "p_error", "precision")

    # P_error vs tiempo de inferencia
    plot_times_for_p_error(directory_path)


if __name__ == "__main__":
    main()
