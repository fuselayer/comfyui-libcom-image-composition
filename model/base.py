"""
Base classes for harmonization models
Provides unified interface for all model types
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseHarmonizationModel(ABC, nn.Module):
    """
    Abstract base class for all harmonization models
    Ensures consistent interface across CDTNet, Harmonizer, iSSAM, etc.
    """
    
    def __init__(self):
        super().__init__()
        self.device = None
    
    @abstractmethod
    def forward(self, image, mask):
        """
        Run harmonization
        
        Args:
            image: Input composite image tensor (B, C, H, W) in [0, 1]
            mask: Object mask tensor (B, 1, H, W) in [0, 1]
            
        Returns:
            Harmonized image tensor (B, C, H, W) in [0, 1]
        """
        pass
    
    @property
    @abstractmethod
    def model_type(self):
        """Return model type identifier (e.g., 'cdtnet', 'harmonizer')"""
        pass
    
    @property
    @abstractmethod
    def requires_fixed_resolution(self):
        """Whether model requires specific input resolution"""
        pass
    
    def to_device(self, device):
        """Move model to device"""
        self.device = device
        return self.to(device)
    
    def set_eval(self):
        """Set model to evaluation mode"""
        self.eval()
        return self


class CDTNetBase(BaseHarmonizationModel):
    """
    Base class for CDTNet-style models (CDTNet, iSSAM)
    Handles normalization and resolution management
    """
    
    def __init__(self):
        super().__init__()
        # ImageNet normalization (used internally)
        self.mean = torch.tensor([.485, .456, .406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], dtype=torch.float32).view(1, 3, 1, 1)
        
    def init_device(self, input_device):
        """Initialize device-specific tensors"""
        if self.device is None:
            self.device = input_device
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)
    
    def normalize(self, tensor):
        """Normalize to ImageNet stats"""
        self.init_device(tensor.device)
        return (tensor - self.mean) / self.std
    
    def denormalize(self, tensor):
        """Denormalize from ImageNet stats"""
        self.init_device(tensor.device)
        return tensor * self.std + self.mean
    
    @property
    def model_type(self):
        return "cdtnet"
    
    @property
    def requires_fixed_resolution(self):
        return False  # Can handle variable resolution


class FilterBasedHarmonizer(BaseHarmonizationModel):
    """
    Base class for filter-based models (Harmonizer, Enhancer)
    """
    
    def __init__(self):
        super().__init__()
        self.input_size = (256, 256)
    
    @property
    def model_type(self):
        return "harmonizer"
    
    @property
    def requires_fixed_resolution(self):
        return True  # Resizes to fixed input_size internally