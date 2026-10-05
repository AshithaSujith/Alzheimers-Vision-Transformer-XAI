Enhanced Alzheimer's Detection Using Vision Transformer and Explainable AI
A deep learning-based multi-class Alzheimer's disease detection project using a Vision Transformer (ViT-Tiny) with CLAHE-based MRI preprocessing and Grad-CAM explainability.
Note: This project is intended for research and educational purposes. It is not a clinical diagnostic system.


Project Overview
This project classifies brain MRI images into four Alzheimer's-related categories using a pretrained Vision Transformer model.
The workflow combines:
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for MRI contrast enhancement
- ViT-Tiny (vit_tiny_patch16_224) for multi-class image classification
- Mixed-precision training for faster GPU training
- Grad-CAM for visual explanation of model predictions
- Gradio for an interactive prediction interface

  
Classes
The model predicts four classes:
- Mild Demented
- Moderate Demented
- Non-Demented
- Very Mild Demented
The class ordering is obtained from the dataset for the training pipeline and is explicitly defined in the XAI inference interface.

Dataset
The training notebook uses the Kaggle dataset:
alzheimers-multiclass-dataset-equal-and-augmented
and reads images from:
combined_images
The dataset itself is not included in this repository.
The images are loaded using torchvision.datasets.ImageFolder.


Data Split
The notebook creates:
- 80% training data
- 10% validation data
- 10% test data
A fixed random seed (42) is used for the split.

Image Preprocessing
CLAHE
CLAHE is applied to the luminance channel of the MRI image in the LAB color space.
Configuration used:
- Clip limit: 1.5
- Tile grid size: (8, 8)
- 
Transformations
For training:
- Resize to 224 × 224
- Random horizontal flip
- Convert to tensor
- Normalize using mean [0.5, 0.5, 0.5]
- Normalize using standard deviation [0.5, 0.5, 0.5]
For validation and testing:
- Resize to 224 × 224
- Convert to tensor
- Normalize using [0.5, 0.5, 0.5]

  
Model
The classification model is:
Vision Transformer (ViT-Tiny)
vit_tiny_patch16_224
Configuration:
- Pretrained: ImageNet
- Number of output classes: 4
- Dropout rate: 0.1
- Optimizer: AdamW
- Learning rate: 1e-4
- Loss function: Cross Entropy Loss
- Batch size: 128
- Epochs: 8
- Mixed precision: Enabled

  
Training
The model is trained using GPU acceleration when available.
Mixed-precision training is implemented using:
- torch.amp.autocast
- GradScaler
The trained model is saved as:
realistic_alz_model.pth
The notebook then evaluates the model on the unseen test split.


Evaluation
The training notebook generates:
- Classification report
- Confusion matrix
- Multi-class ROC curves
- AUC values for each class
The classification report includes:
- Precision
- Recall
- F1-score
- Support
The repository does not include a hard-coded performance value because the notebook computes the final metrics when it is executed.


Explainable AI
The second notebook provides a Grad-CAM-based explanation workflow for the trained Vision Transformer.
The implementation targets the final Vision Transformer attention block:
model.blocks[-1].norm1
The gradients and feature representations from this layer are used to generate a class activation map.
The resulting activation map is:
1. Converted into a 2D 14 × 14 patch representation
2. Resized to 224 × 224
3. Converted into a heatmap
4. Overlaid on the original MRI image
This provides a visual indication of the image regions contributing to the model's prediction.


Interactive Demo
The XAI notebook uses Gradio to create an interactive interface.
Users can:
1. Upload a brain MRI image
2. Receive prediction confidence for all four classes
3. View the predicted class
4. View the Grad-CAM visualization
The interface is titled:
Alzheimer's Multi-Class Diagnosis (ViT + Grad-CAM)


Repository Structure
Alzheimers-Vision-Transformer-XAI/
│
├── model.ipynb       # Model training and evaluation
├── xai11.ipynb       # Grad-CAM and Gradio inference interface
└── README.md         # Project documentation


Technologies Used
- Python
- PyTorch
- Torchvision
- timm
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- PIL
- Gradio
- Vision Transformer (ViT)
- Grad-CAM
- Kaggle


Notebooks
model.ipynb
Contains:
- Dataset loading
- CLAHE preprocessing
- Train/validation/test splitting
- ViT-Tiny initialization
- Model training
- Test evaluation
- Classification report
- Confusion matrix
- ROC/AUC analysis
xai11.ipynb
Contains:
- Trained model loading
- ViT inference
- Prediction confidence
- Custom Grad-CAM implementation
- Heatmap generation
- Gradio interface

  
How to Run
1. Install dependencies
pip install torch torchvision timm opencv-python numpy matplotlib seaborn scikit-learn pillow tqdm gradio
2. Prepare the dataset
Download the required Alzheimer's MRI dataset from Kaggle and update the dataset path in model.ipynb.
3. Train and evaluate
Run:
model.ipynb
This trains the ViT-Tiny model and performs evaluation on the test split.
4. Run the XAI demo
Update the trained model path in xai11.ipynb and run the notebook.
The Gradio interface allows an MRI image to be uploaded for prediction and Grad-CAM visualization.


Limitations
- The project is based on a specific Kaggle MRI dataset.
- Performance depends on the dataset and train/validation/test split.
- Grad-CAM for Vision Transformers requires careful interpretation of transformer token representations.
- The system is a research prototype and should not be used as a standalone medical diagnostic tool.

  
Author
Ashitha P. Sujith
MSc Computer Science
