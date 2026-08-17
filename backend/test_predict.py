import torch

from torchvision import datasets, transforms

from predict import predict_image


# ============================================================
# 1. LOAD MNIST TEST DATA
# ============================================================

dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


# ============================================================
# 2. SELECT ONE IMAGE
# ============================================================

index = 0

image, actual_label = dataset[index]


# ============================================================
# 3. SAVE IMAGE
# ============================================================

image_path = "./test_digit.png"

transforms.ToPILImage()(image).save(
    image_path
)


print(
    f"Actual digit: {actual_label}"
)


# ============================================================
# 4. PREDICT
# ============================================================

result = predict_image(
    image_path
)


# ============================================================
# 5. DISPLAY RESULT
# ============================================================

print("\nPrediction Result")
print("=" * 40)

print(
    f"Predicted digit : "
    f"{result['digit']}"
)

print(
    f"Confidence       : "
    f"{result['confidence'] * 100:.2f}%"
)

print("\nAll probabilities:")

for digit, probability in enumerate(
    result["probabilities"]
):

    print(
        f"{digit}: "
        f"{probability * 100:.2f}%"
    )