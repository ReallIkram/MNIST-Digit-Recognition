import os

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = "./model/mnist_cnn.pth"


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# 3. CNN MODEL
# ============================================================

class MNIST_CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# 4. CREATE MODEL
# ============================================================

model = MNIST_CNN().to(device)


# ============================================================
# 5. LOAD TRAINED WEIGHTS
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


# Put model into evaluation mode
model.eval()

print("Trained model loaded successfully.")


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    # Convert image to tensor
    transforms.ToTensor(),

    # Same normalization used during training
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("L")


    # --------------------------------------------------------
    # Resize to MNIST size
    #
    # MNIST images = 28 × 28
    # --------------------------------------------------------

    image = image.resize(
        (28, 28)
    )


    # --------------------------------------------------------
    # Convert image to tensor
    # --------------------------------------------------------

    image_tensor = transform(
        image
    )


    # --------------------------------------------------------
    # Add batch dimension
    #
    # Before:
    #
    # 1 × 28 × 28
    #
    # After:
    #
    # 1 × 1 × 28 × 28
    # --------------------------------------------------------

    image_tensor = image_tensor.unsqueeze(
        0
    )


    # Move image to GPU/CPU

    image_tensor = image_tensor.to(
        device
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )


        # Convert output scores to probabilities

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # Get highest probability

        confidence, predicted_class = (
            torch.max(
                probabilities,
                dim=1
            )
        )


    # --------------------------------------------------------
    # Convert tensors to normal Python values
    # --------------------------------------------------------

    digit = predicted_class.item()

    confidence = confidence.item()


    # --------------------------------------------------------
    # Get probability for every digit
    # --------------------------------------------------------

    all_probabilities = (
        probabilities[0]
        .cpu()
        .tolist()
    )


    return {
        "digit": digit,
        "confidence": confidence,
        "probabilities": all_probabilities
    }


# ============================================================
# 8. TEST PREDICTION
# ============================================================

if __name__ == "__main__":

    print("\nModel is ready for prediction.")

    print(
        "Use predict_image('path/to/image.png') "
        "to classify an image."
    )