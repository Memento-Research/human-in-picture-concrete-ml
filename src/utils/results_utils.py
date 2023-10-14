def export_results(image_size: int, n_bits: float, p_error: float,
                   clear_precision: float, fhe_precision: float,
                   times: [],
                   use_sim: bool, n_times: int):
    export(image_size, n_bits, p_error, clear_precision, fhe_precision, "times", times, use_sim, n_times)


def export_losses(image_size: int, n_bits: float, p_error: float,
                  clear_precision: float, fhe_precision: float,
                  losses: [],
                  use_sim: bool, n_times: int):
    export(image_size, n_bits, p_error, clear_precision, fhe_precision, "losses", losses, use_sim, n_times)


def export(image_size: int, n_bits: float, p_error: float,
           clear_precision: float, fhe_precision: float,
           data_header: str, data: [],
           use_sim: bool, n_times: int):
    mode = "sim" if use_sim else "fhe"
    p_error = int(p_error * 10)
    file_name = f"{data_header}_{n_times}_{mode}_{image_size}px_{n_bits}b_{p_error}.txt"

    with open(f"results/{data_header}/{file_name}", "w") as f:
        f.write(f"Image size: {image_size}\n")
        f.write(f"Number of bits: {n_bits}\n")
        f.write(f"Probability of error: {p_error}\n")
        f.write(f"Clear precision: {clear_precision}\n")
        f.write(f"FHE precision: {fhe_precision}\n")
        f.write(f"{data_header}:\n")
        for row in data:
            f.write(f"{row}\n")
