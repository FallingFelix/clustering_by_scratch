import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt


def show_keypoints(img, keypoints, title, save_path):
    """Save SIFT keypoints as red dots on image."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    for kp in keypoints:
        plt.plot(kp.pt[0], kp.pt[1], 'r.', markersize=2)
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")


def show_matched_features(img1, img2, matched_pairs, save_path):
    """Save top matched keypoint pairs with blue lines."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1, :] = img1
    vis[:h2, w1:w1 + w2, :] = img2

    for pt1, pt2 in matched_pairs:
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]) + w1, int(pt2[1])
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.imshow(vis_rgb)
    plt.title("Top 10% Matched Features")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")


def extract_and_match_features(img1, img2, top_k_ratio=0.1):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Save keypoints
    show_keypoints(img1, kp1, "SIFT Keypoints - Image 1", "keypoints_img1.png")
    show_keypoints(img2, kp2, "SIFT Keypoints - Image 2", "keypoints_img2.png")

    # Use BFMatcher for fast nearest-neighbor matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    cv_matches = bf.match(des1, des2)

    print(f"Total matches found: {len(cv_matches)}")

    # Sort and prune to top 10%
    cv_matches = sorted(cv_matches, key=lambda x: x.distance)
    top_n = max(1, int(len(cv_matches) * top_k_ratio))
    top_matches = cv_matches[:top_n]

    print(f"Top 10% matches retained: {top_n}")

    matched_pairs = [(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt) for m in top_matches]
    show_matched_features(img1, img2, matched_pairs, "matched_points.png")
    return matched_pairs


if __name__ == "__main__":
    img_path1 = 'SIFT1_img.jpg'
    img_path2 = 'SIFT2_img.jpg'

    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    if img1 is None or img2 is None:
        print("NO IMAGES FOUND")
        sys.exit(1)

    matched_keypoints = extract_and_match_features(img1, img2)
    print(f"Final matched keypoint count: {len(matched_keypoints)}")

