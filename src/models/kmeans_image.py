from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[2]

IMAGE = ROOT / "src" / "data" / "input_image.jpg"

OUT = ROOT / "reports" / "figures"

OUT.mkdir(
    parents=True,
    exist_ok=True
)


def run():

    print("\n" + "=" * 50)
    print("--- K-MEANS IMAGE SEGMENTATION ---")

    print(
        f"Loading image: {IMAGE}"
    )

    img = (
        Image.open(IMAGE)
        .convert("RGB")
        .resize((300, 300))
    )

    arr = np.array(img)

    print(
        f"Image shape: {arr.shape}"
    )

    X = arr.reshape(
        -1,
        3
    ).astype(float)

    ks = [
        2,
        4,
        6,
        8
    ]

    inertias = []
    segments = []

    for k in ks:

        print(
            f"\nRunning K-Means with K={k}..."
        )

        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42
        )

        labels = model.fit_predict(X)

        segmented = (
            model.cluster_centers_[labels]
            .reshape(arr.shape)
            .astype("uint8")
        )

        inertias.append(
            model.inertia_
        )

        segments.append(
            segmented
        )

        output_file = (
            OUT /
            f"segmented_k{k}.png"
        )

        Image.fromarray(
            segmented
        ).save(
            output_file
        )

        print(
            f"Saved: {output_file}"
        )


    print("\n" + "=" * 50)
    print("--- SEGMENTATION COMPARISON ---")

    fig, ax = plt.subplots(
        1,
        5,
        figsize=(18, 4)
    )

    ax[0].imshow(arr)

    ax[0].set_title(
        "Original"
    )

    ax[0].axis("off")

    for a, k, segment in zip(
        ax[1:],
        ks,
        segments
    ):

        a.imshow(segment)

        a.set_title(
            f"K={k}"
        )

        a.axis("off")

    plt.tight_layout()

    comparison_file = (
        OUT /
        "image_segmentation_comparison.png"
    )

    plt.savefig(
        comparison_file,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {comparison_file}"
    )


    print("\n" + "=" * 50)
    print("--- ELBOW CURVE ---")

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        ks,
        inertias,
        marker="o"
    )

    plt.xlabel(
        "K"
    )

    plt.ylabel(
        "Inertia"
    )

    plt.title(
        "Elbow Curve - Image Segmentation"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.6
    )

    plt.tight_layout()

    elbow_file = (
        OUT /
        "image_segmentation_elbow.png"
    )

    plt.savefig(
        elbow_file,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {elbow_file}"
    )

    print("\n" + "=" * 50)

    print(
        "Week 11 K-Means image segmentation "
        "execution complete."
    )


if __name__ == "__main__":
    run()