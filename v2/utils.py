from matplotlib import pyplot as plt
import tensorflow as tf


def show_images(images, title="Noisy Images at Timestep"):
    B, H, W, C = images.shape
    # Convert tensor to numpy (if needed) and select the first few images
    images = images.cpu().numpy()
    num_channels = images.shape[-1]

    # Create subplots for the images
    fig, axes = plt.subplots(1, B, figsize=(12, 3))
    fig.suptitle(f"{title}", fontsize=16)

    for i, ax in enumerate(axes):
        img = images[i]

        # Handle grayscale or RGB images
        if num_channels == 1:  # Grayscale
            img = img.squeeze()  # Remove single channel dimension
            ax.imshow(img, cmap="gray")
        else:  # RGB
            ax.imshow(img)

        ax.axis("off")  # Turn off axis for clean display

    plt.show()
