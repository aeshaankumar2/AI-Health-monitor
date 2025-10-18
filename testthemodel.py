import torch
from torchvision import models, transforms
from PIL import Image
import os

MODEL_PATH = os.path.expanduser("~/outputs/checkpoint_best.pth")
IMG_PATH = os.path.expanduser("~/Desktop/dataset_processed/train/abrasion/000005.jpg") 
CLASS_NAMES = ['abrasion', 'bruise', 'burn', 'infection', 'laceration', 'rash']

device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = Image.open(IMG_PATH).convert("RGB")
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    pred_idx = probs.argmax().item()

print("\nPrediction Results:")
for i, (cls, p) in enumerate(zip(CLASS_NAMES, probs)):
    print(f"{cls:<12}: {p.item()*100:.2f}%")
print(f"\nPredicted Class: {CLASS_NAMES[pred_idx].upper()}")
