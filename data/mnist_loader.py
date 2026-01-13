import urllib.request  # For downloading files from the internet
import gzip            # For unzipping .gz compressed files
import os              # For file and directory operations

def download_mnist():
    """Download MNIST dataset if not already present"""
    
    # The website where MNIST is hosted (using HTTPS)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
    # The four files we need:
    # - train-images: 60,000 training images
    # - train-labels: 60,000 training labels
    # - t10k-images: 10,000 test images
    # - t10k-labels: 10,000 test labels
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    # Create data/mnist directory if it doesn't exist
    if not os.path.exists("data/mnist"):
        os.makedirs("data/mnist")
    
    # Download each file
    for filename in files:
        filepath = f"data/mnist/{filename}"
        
        # Only download if file doesn't already exist
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            # Download from URL and save to filepath
            urllib.request.urlretrieve(base_url + filename, filepath)
            print(f"Downloaded {filename}")
        else:
            print(f"{filename} already exists")


def load_images(filename):
    """Load MNIST images from gzipped file"""
    
    # Open the gzipped file for reading bytes
    with gzip.open(filename, 'rb') as f:
        
        # The file starts with 16 bytes of header info (magic number, counts)
        # We skip this - we just want the pixel data
        f.read(16)
        
        # Read all remaining bytes (the actual pixel values)
        data = f.read()
        
        # Convert bytes to list of integers (each pixel is 0-255)
        images = list(data)
        
        # Each image is 28x28 = 784 pixels
        # Calculate how many images we have
        num_images = len(images) // 784
        
        # Split into individual images
        result = []
        for i in range(num_images):
            # Extract 784 pixels for image i
            image = images[i * 784 : (i + 1) * 784]
            
            # Normalize pixel values from 0-255 to 0-1
            # This helps with training (smaller numbers = more stable)
            image = [p / 255.0 for p in image]
            
            result.append(image)
        
        return result


def load_labels(filename):
    """Load MNIST labels from gzipped file"""
    
    # Open the gzipped file
    with gzip.open(filename, 'rb') as f:
        
        # Labels file has 8 bytes of header
        f.read(8)
        
        # Read all label bytes (each label is one byte: 0-9)
        data = f.read()
        
        # Convert to list of integers
        return list(data)


def load_mnist():
    """Load full MNIST dataset - returns training and test sets"""
    
    # First, make sure data is downloaded
    download_mnist()
    
    # Load training data (60,000 images and labels)
    train_images = load_images("data/mnist/train-images-idx3-ubyte.gz")
    train_labels = load_labels("data/mnist/train-labels-idx1-ubyte.gz")
    
    # Load test data (10,000 images and labels)
    test_images = load_images("data/mnist/t10k-images-idx3-ubyte.gz")
    test_labels = load_labels("data/mnist/t10k-labels-idx1-ubyte.gz")
    
    return train_images, train_labels, test_images, test_labels

def print_image(image, label):
    """Print an image as ASCII art"""
    print(f"\nLabel: {label}")
    print("-" * 30)
    
    # Image is 784 pixels in a flat list
    # We need to print it as 28 rows of 28 pixels
    for row in range(28):
        line = ""
        for col in range(28):
            # Get pixel value (0 to 1)
            pixel = image[row * 28 + col]
            
            # Convert to ASCII character
            if pixel > 0.75:
                line += "█"  # Very bright
            elif pixel > 0.5:
                line += "▓"  # Bright
            elif pixel > 0.25:
                line += "░"  # Dim
            else:
                line += " "  # Dark (background)
        print(line)


if __name__ == "__main__":
    print("Loading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    print(f"\nTraining set: {len(train_images)} images")
    print(f"Test set: {len(test_images)} images")
    
    # Show first 5 images
    print("\n--- First 5 training images ---")
    for i in range(5):
        print_image(train_images[i], train_labels[i])