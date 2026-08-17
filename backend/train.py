import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

DATA_DIR = "./data"
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "mnist_cnn.pth")


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MNIST CNN TRAINING")
print("=" * 60)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA version    : {torch.version.cuda}")
print(f"Using device    : {device}")

if torch.cuda.is_available():

    print(
        f"GPU             : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU memory      : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

else:

    print("WARNING: CUDA is not available.")
    print("Training will use CPU.")

print("=" * 60)


# ============================================================
# 3. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    # Convert PIL image into PyTorch tensor
    transforms.ToTensor(),

    # Normalize MNIST pixels
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


# ============================================================
# 4. LOAD MNIST DATASET
# ============================================================

print("\nLoading MNIST dataset...")

train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

print(
    f"Training samples: {len(train_dataset)}"
)

print(
    f"Testing samples : {len(test_dataset)}"
)


# ============================================================
# 5. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 6. CNN MODEL
# ============================================================

class MNIST_CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # First convolution
            #
            # Input:
            # 1 x 28 x 28
            #
            # Output:
            # 32 x 28 x 28
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # ------------------------------------------------
            # Max Pooling
            #
            # 28 x 28
            #     ↓
            # 14 x 14
            # ------------------------------------------------

            nn.MaxPool2d(
                kernel_size=2
            ),

            # ------------------------------------------------
            # Second convolution
            #
            # 32 channels
            #     ↓
            # 64 channels
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # ------------------------------------------------
            # Second pooling
            #
            # 14 x 14
            #     ↓
            # 7 x 7
            # ------------------------------------------------

            nn.MaxPool2d(
                kernel_size=2
            )
        )

        self.classifier = nn.Sequential(

            # ------------------------------------------------
            # 64 feature maps × 7 × 7
            #
            # 64 × 7 × 7 = 3136
            # ------------------------------------------------

            nn.Flatten(),

            nn.Linear(
                64 * 7 * 7,
                128
            ),

            nn.ReLU(),

            # Helps reduce overfitting
            nn.Dropout(
                0.5
            ),

            # ------------------------------------------------
            # 10 output classes
            #
            # 0 1 2 3 4 5 6 7 8 9
            # ------------------------------------------------

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
# 7. CREATE MODEL
# ============================================================

model = MNIST_CNN().to(device)

print("\nCNN model created successfully.")


# ============================================================
# 8. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 9. OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 10. TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    correct = 0
    total = 0

    for images, labels in train_loader:

        # Move images to CPU/GPU
        images = images.to(
            device,
            non_blocking=True
        )

        # Move labels to CPU/GPU
        labels = labels.to(
            device,
            non_blocking=True
        )

        # Remove gradients from previous step
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate error
        loss = criterion(
            outputs,
            labels
        )

        # Calculate gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Track loss
        total_loss += loss.item()

        # Get predicted digit
        predictions = outputs.argmax(
            dim=1
        )

        # Count correct predictions
        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = (
        total_loss / len(train_loader)
    )

    accuracy = (
        100.0 * correct / total
    )

    return average_loss, accuracy


# ============================================================
# 11. TESTING / EVALUATION FUNCTION
# ============================================================

def evaluate():

    # Put model into evaluation mode
    model.eval()

    correct = 0
    total = 0

    # Disable gradient calculation
    # because we are not training here.
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # Prediction
            outputs = model(images)

            # Get highest scoring class
            predictions = outputs.argmax(
                dim=1
            )

            # Count correct predictions
            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = (
        100.0 * correct / total
    )

    return accuracy


# ============================================================
# 12. TRAINING LOOP
# ============================================================

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    test_accuracy = evaluate()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )


# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)


# ============================================================
# 14. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Final test accuracy: "
    f"{test_accuracy:.2f}%"
)

print(
    f"Model saved to: "
    f"{MODEL_PATH}"
)

print("=" * 60)