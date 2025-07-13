
import numpy as np
import matplotlib.pyplot as plt
import cv2

def manual_kmeans(data, k, max_iter=300, runs=10):
    best_sse = float('inf')
    best_labels = None
    best_centers = None

    for _ in range(runs):
        indices = np.random.choice(len(data), k, replace=False)
        centers = data[indices]

        for _ in range(max_iter):
            distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centers = np.array([data[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
                                    for j in range(k)])

            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        sse = sum(np.linalg.norm(data[i] - centers[labels[i]])**2 for i in range(len(data)))
        if sse < best_sse:
            best_sse = sse
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers, best_sse

def visualize_manual_kmeans_2d(data, k_values):
    fig, axes = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 5))
    for idx, k in enumerate(k_values):
        labels, centers, sse = manual_kmeans(data, k)
        print(f"K={k}, SSE={sse:.2f}")
        ax = axes[idx]
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)
        ax.set_title(f"K={k}, SSE={sse:.2f}")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(True)
    plt.tight_layout()
    plt.savefig("kmeans_comparison_manual.png")
    plt.show()

# ============================
# Color Quantization using Manual K-Means

def quantize_image_manual_kmeans(image, k):
    h, w, c = image.shape
    flat_img = image.reshape((-1, 3))

    labels, centers, _ = manual_kmeans(flat_img, k)
    quantized_img = centers[labels].reshape((h, w, 3)).astype(np.uint8)
    return quantized_img

def show_all_image_comparisons():
    orig1 = cv2.imread("Kmean_img1.jpg")
    orig2 = cv2.imread("Kmean_img2.jpg")
    orig1 = cv2.cvtColor(orig1, cv2.COLOR_BGR2RGB)
    orig2 = cv2.cvtColor(orig2, cv2.COLOR_BGR2RGB)

    quant1 = quantize_image_manual_kmeans(orig1, 5)
    quant2 = quantize_image_manual_kmeans(orig2, 10)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(orig1)
    axes[0, 0].set_title("Original Image 1")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(quant1)
    axes[1, 0].set_title("Quantized Image 1 (K=5)")
    axes[1, 0].axis("off")

    axes[0, 1].imshow(orig2)
    axes[0, 1].set_title("Original Image 2")
    axes[0, 1].axis("off")

    axes[1, 1].imshow(quant2)
    axes[1, 1].set_title("Quantized Image 2 (K=10)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("kmeans_all_images_comparison.png")
    plt.show()

if __name__ == "__main__":
    data_2d = np.loadtxt("510_cluster_dataset.txt")
    visualize_manual_kmeans_2d(data_2d, [2, 3, 4])

    show_all_image_comparisons()
