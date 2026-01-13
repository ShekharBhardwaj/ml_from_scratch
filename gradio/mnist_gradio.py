import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import numpy as np

# ============================================================
# STEP 1: DEFINE THE MODEL
# ============================================================

class MNISTNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x

# ============================================================
# STEP 2: LOAD TRAINED WEIGHTS
# ============================================================

model = MNISTNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

print("Model loaded!")

# ============================================================
# STEP 3: PREDICTION FUNCTION
# ============================================================

def predict(image):
    """
    Take user's drawing and predict the digit.
    """
    
    if image is None:
        return {str(i): 0.0 for i in range(10)}
    
    # Handle dict format from sketchpad (has 'composite' key)
    if isinstance(image, dict):
        if 'composite' in image:
            image = image['composite']
        else:
            return {str(i): 0.0 for i in range(10)}
    
    # Convert to PIL Image
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype('uint8'))
    elif isinstance(image, Image.Image):
        img = image
    else:
        return {str(i): 0.0 for i in range(10)}
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Invert (MNIST is white on black)
    img_array = 255 - img_array
    
    # Normalize to 0-1
    img_array = img_array / 255.0
    
    # Convert to tensor [1, 1, 28, 28]
    tensor = torch.tensor(img_array, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        probs = probabilities[0].tolist()
        result = {str(i): probs[i] for i in range(10)}
    
    return result

# ============================================================
# STEP 4: CREATE SIMPLE INTERFACE
# ============================================================

demo = gr.Interface(
    fn=predict,
    inputs="sketchpad",      # Simple sketchpad
    outputs=gr.Label(num_top_classes=10),
    title="MNIST Digit Recognizer",
    description="Draw a digit (0-9) and click Submit"
)

# ============================================================
# STEP 5: LAUNCH
# ============================================================

print("Starting app...")
demo.launch()