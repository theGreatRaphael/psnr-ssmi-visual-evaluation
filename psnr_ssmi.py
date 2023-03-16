from math import log10, sqrt
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssmi_fun
import matplotlib.pyplot as plt


def PSNR(original: np.ndarray, blurred: np.ndarray) -> float:
    mse = np.mean((original - blurred) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def gaussian_noise(img: np.ndarray, ksize: tuple[int, int] = [31, 31]) -> np.ndarray:
    return cv2.GaussianBlur(img, ksize, 0)


def SSMI(original: np.ndarray, blurred: np.ndarray) -> float:
    return ssmi_fun(original, blurred, channel_axis=2, full=False)


def plot_results(original: np.ndarray, comparisons: list[np.ndarray], ksize: list[float], psnrs: list[float], ssmis: list[float]) -> None:
    f, axarr = plt.subplots(1, len(comparisons) + 1, figsize=(20, 10))
    axarr[0].imshow(original)
    axarr[0].title.set_text(
        f"Original image, PSNR: {psnrs[0]: .3f}, SSMI: {ssmis[0]: .3f}")

    for i in range(len(comparisons)):
        axarr[i + 1].imshow(comparisons[i])
        axarr[i + 1].title.set_text(
            f"Kernel Size: {ksize[i]}, PSNR: {psnrs[1 + i]: .3f}, SSMI: {ssmis[1 + i]: .3f}")

    plt.show()

    f.savefig("nice_plots.png", dpi=f.dpi)


def run_experiment(img: np.ndarray, ksizes: list[int] = [3, 9, 21, 51, 71, 101, 151, 201]) -> None:
    psnr_res = []
    ssmi_res = []

    for ks in ksizes:
        blurred = gaussian_noise(img.copy(), ksize=(ks, ks))
        psnr_res.append(PSNR(img, blurred))
        ssmi_res.append(SSMI(img, blurred))

    f, axarr = plt.subplots(1, 2, figsize=(10, 10))

    axarr[0].set_xlabel("Gaussian kernel size")
    axarr[0].set_ylabel("PSNR score")
    axarr[0].title.set_text(
        f"PSNR scores over different gaussian kernel sizes")
    axarr[0].plot(ksizes, psnr_res, label="PSNR", color="green")

    axarr[1].set_xlabel("Gaussian kernel size")
    axarr[1].set_ylabel("SSMI score")
    axarr[1].title.set_text(
        f"SSMI scores over different gaussian kernel sizes")
    axarr[1].plot(ksizes, ssmi_res, label="SSMI", color="blue")

    plt.show()
    f.savefig('experiment.png', dpi=f.dpi)


def main():
    original = cv2.imread("test_img.jpg")
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    blurred_1 = gaussian_noise(original.copy())
    blurred_2 = gaussian_noise(original.copy(), ksize=(101, 101))

    psnr_same_img = PSNR(original, original)
    ssmi_same_img = SSMI(original, original)

    print(
        f"Sanity check same img -> PSNR: {psnr_same_img}, SSMI: {ssmi_same_img}")

    psnr_1 = PSNR(original, blurred_1)
    ssmi_1 = SSMI(original, blurred_1)

    psnr_2 = PSNR(original, blurred_2)
    ssmi_2 = SSMI(original, blurred_2)

    plot_results(
        original=original,
        comparisons=[blurred_1, blurred_2],
        ksize=[31, 101],
        psnrs=[psnr_same_img, psnr_1, psnr_2],
        ssmis=[ssmi_same_img, ssmi_1, ssmi_2]
    )

    run_experiment(original)


if __name__ == "__main__":
    main()
