"""
Harmony Score Evaluation - No ground truth needed
Based on: https://github.com/bcmi/libcom
Uses BargainNet to predict harmony score (0-1, higher = better)
"""

import torch
import numpy as np
import sys
import os


# Add model/ to path (so 'from libcom.utils...' works)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..')
if MODEL_PATH not in sys.path:
    sys.path.insert(0, MODEL_PATH)

from libcom.harmony_score import HarmonyScoreModel  # Changed this line!


class HarmonyScorer:
    """
    Predicts harmonization quality WITHOUT ground truth
    
    Uses BargainNet to compare foreground/background style consistency.
    Returns harmony score 0-1 where higher = more harmonious.
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize harmony scorer
        
        Args:
            device: 'cuda' or 'cpu' or cuda device index (0, 1, etc.)
        """
        self.device = device
        self.model = None
    
    def load_model(self):
        """Lazy load model (downloads pretrained weights if needed)"""
        if self.model is None:
            print("Loading HarmonyScoreModel (BargainNet)...")
            self.model = HarmonyScoreModel(device=self.device, model_type='BargainNet')
        return self.model
    
    def tensor_to_numpy(self, tensor, is_mask=False):
        """
        Convert ComfyUI tensor to numpy array for HarmonyScoreModel
        
        Args:
            tensor: ComfyUI format - (B, H, W, C) for image or (B, H, W) for mask
            is_mask: If True, converts to grayscale mask
        
        Returns:
            numpy array in format HarmonyScoreModel expects
        """
        # Move to CPU first
        tensor = tensor.cpu()
        
        if is_mask:
            # Mask: (B, H, W) → (H, W) in [0, 255]
            if tensor.dim() == 3:
                tensor = tensor[0]  # Take first batch
            np_array = (tensor.numpy() * 255).astype(np.uint8)
        else:
            # Image: (B, H, W, C) → (H, W, C) in [0, 255]
            if tensor.dim() == 4:
                tensor = tensor[0]  # Take first batch
            np_array = (tensor.numpy() * 255).astype(np.uint8)
            # Convert RGB to BGR (opencv format)
            np_array = np_array[:, :, ::-1]
        
        return np_array
    
    def score(self, composite_image, mask):
        """
        Score harmony of composite image
        
        Args:
            composite_image: Tensor (B, C, H, W) in [0, 1] or numpy/filepath
            mask: Tensor (B, 1, H, W) in [0, 1] or numpy/filepath
        
        Returns:
            harmony_score: Float 0-1 (higher = better harmony)
        """
        model = self.load_model()
        
        # Convert tensors to numpy if needed
        if isinstance(composite_image, torch.Tensor):
            composite_image = self.tensor_to_numpy(composite_image, is_mask=False)
            
            print(f"Converted image shape: {composite_image.shape}, dtype: {composite_image.dtype}, range: [{composite_image.min()}, {composite_image.max()}]")
        
        if isinstance(mask, torch.Tensor):
            mask = self.tensor_to_numpy(mask, is_mask=True)
        
        # Get harmony score
        score = model(composite_image, mask)
        
        return float(score)
    
    def score_batch(self, images, masks):
        """
        Score multiple images
        
        Args:
            images: List of images or batch tensor
            masks: List of masks or batch tensor
        
        Returns:
            List of harmony scores
        """
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            # Batch tensor - split into list
            images = [images[i] for i in range(images.size(0))]
            masks = [masks[i] for i in range(masks.size(0))]
        
        scores = []
        for img, mask in zip(images, masks):
            score = self.score(img, mask)
            scores.append(score)
        
        return scores