import io
import os

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from preprocess import preprocess_image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


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

print("=" * 60)
print("MNIST API")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print("=" * 60)


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
# 5. LOAD TRAINED MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Trained MNIST model loaded successfully.")


# ============================================================
# 6. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


# ============================================================
# 7. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="MNIST Digit Recognition API",
    description="PyTorch CNN API for handwritten digit recognition",
    version="1.0.0"
)


# ============================================================
# 8. CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# 9. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "MNIST Digit Recognition API is running",
        "device": str(device)
    }


# ============================================================
# 10. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }


# ============================================================
# 11. PREDICT ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    image_bytes = await file.read()


    # --------------------------------------------------------
    # Convert bytes into PIL image
    # --------------------------------------------------------

    image = Image.open(
        io.BytesIO(image_bytes)
    )
    image = preprocess_image(image)


    # --------------------------------------------------------
    # Convert image to tensor
    # --------------------------------------------------------

    image_tensor = transform(
        image
    )


    # --------------------------------------------------------
    # Add batch dimension
    #
    # 1 × 28 × 28
    #       ↓
    # 1 × 1 × 28 × 28
    # --------------------------------------------------------

    image_tensor = image_tensor.unsqueeze(
        0
    )


    # --------------------------------------------------------
    # Move tensor to GPU / CPU
    # --------------------------------------------------------

    image_tensor = image_tensor.to(
        device
    )


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )


    # --------------------------------------------------------
    # Convert result to Python values
    # --------------------------------------------------------

    digit = predicted_class.item()

    confidence = confidence.item()


    # --------------------------------------------------------
    # Get probabilities for all digits
    # --------------------------------------------------------

    all_probabilities = (
        probabilities[0]
        .cpu()
        .tolist()
    )


    # --------------------------------------------------------
    # Return JSON response
    # --------------------------------------------------------

    return {

        "success": True,

        "digit": digit,

        "confidence": confidence,

        "probabilities": {
            str(i): all_probabilities[i]
            for i in range(10)
        }
    }