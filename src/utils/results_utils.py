def export_times(image_size: int, times: []):
    with open(f"results/results_{image_size}px.txt", "w") as f:
        f.write(f"Image size: {image_size}\n")
        for time in times:
            f.write(f"{time}\n")
