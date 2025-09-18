import numpy as np
import matplotlib.pyplot as plt
import cv2

# Manual K-Means 
def kmeans(data, k, max_iter=300, runs=10):
    best_sse = float('inf')
    best_labels = None
    best_centers = None

    print("This will take a while")

    for _ in range(runs):
        indices = np.random.choice(len(data), k, replace=False)
        centers = data[indices]

        for _ in range(max_iter):
            distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centers = np.array([
                data[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
                for j in range(k)
            ])

            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        sse = sum(np.linalg.norm(data[i] - centers[labels[i]])**2 for i in range(len(data)))
        if sse < best_sse:
            best_sse = sse
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers, best_sse

# Combine 2D Scatter and Image Transformations
def kmeans_image(image, k):
    h, w, c = image.shape
    flat_img = image.reshape((-1, 3))

    labels, centers, _ = kmeans(flat_img, k)
    quantized_img = centers[labels].reshape((h, w, 3)).astype(np.uint8)
    return quantized_img

def plot_scatter_and_image(data_2d, image1_path, image2_path):
    # Apply KMeans to data
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Scatter plots for K=2, 3, 4
    for idx, k in enumerate([2, 3, 4]):
        labels, centers, sse = kmeans(data_2d, k)
        ax = axes[0, idx]
        ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', s=10)
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)
        ax.set_title(f"2D KMeans Clustering (K={k}, SSE={sse:.2f})")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(True)

    # Hide the fourth subplot in top row
    axes[0, 3].axis('off')

    # Image transformations
    orig1 = cv2.imread(image1_path)
    orig2 = cv2.imread(image2_path)
    orig1 = cv2.cvtColor(orig1, cv2.COLOR_BGR2RGB)
    orig2 = cv2.cvtColor(orig2, cv2.COLOR_BGR2RGB)

    quant5_img1 = kmeans_image(orig1, 5)
    print("Quantized Image 1 with K=5 DONE")
    quant10_img1= kmeans_image(orig1, 10)
    print("Quantized Image 1 with K=10 DONE")
    quant5_img2 = kmeans_image(orig2, 5)
    print("Quantized Image 2 with K=5 DONE")
    quant10_img2 = kmeans_image(orig2, 10)
    print("Quantized Image 2 with K=10 DONE")
    print("All quantized images DONE")

    # Image visualizations in bottom row
    axes[1, 0].imshow(quant5_img1)
    axes[1, 0].set_title("Quantized Image 1 (K=5)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(quant10_img1)
    axes[1, 1].set_title("Quantized Image 1 (K=10)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(quant5_img2)
    axes[1, 2].set_title("Quantized Image 2 (K=5)")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(quant10_img2)
    axes[1, 3].set_title("Quantized Image 2 (K=10)")
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.savefig("final_part_two_image.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    data_2d = np.loadtxt("510_cluster_dataset.txt")
    plot_scatter_and_image(data_2d, "Kmean_img1.jpg", "Kmean_img2.jpg")