## **ResNet50 for Face Embeddings**

**ResNet50** is a deep convolutional neural network widely used for feature extraction and image recognition tasks. In this project, we use it to generate **face embeddings** for recognition.

---

### Key Features

- **Deep Residual Learning**: Introduces “skip connections” to solve vanishing gradient problems in very deep networks.  
- **50 Layers**: Composed of convolutional layers, batch normalization, and identity shortcuts.  
- **Pre-trained Models**: Can be loaded with weights trained on ImageNet for transfer learning.  
- **Embeddings for Faces**: Outputs a fixed-length vector (2048-dimensional in pooling mode) representing a face’s features.

---

### Why ResNet50 for Face Recognition?

- Converts a face image into a **feature vector** (embedding).  
- Similar faces have embeddings that are **close in Euclidean space**.  
- Enables comparison of unknown faces with a database of known embeddings.  
- Works well with MTCNN for real-time detection and recognition.

---

### How We Use It

1. **Crop Face**: Use MTCNN to detect and crop the face from the image.  
2. **Resize**: Resize cropped face to 224×224 pixels.  
3. **Preprocess**: Apply `preprocess_input` to normalize pixels.  
4. **Generate Embedding**:

```python
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load ResNet50 model without top layer (feature extractor)
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Preprocess face image
face_img = image.load_img("face.jpg", target_size=(224, 224))
face_array = image.img_to_array(face_img)
face_array = np.expand_dims(face_array, axis=0)
face_array = preprocess_input(face_array)

# Generate embedding
embedding = model.predict(face_array)[0]  # 2048-d vector
