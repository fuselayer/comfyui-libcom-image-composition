"""
Evaluation metrics for image harmonization
Based on: https://github.com/bcmi/CDTNet-High-Resolution-Image-Harmonization
"""

import torch
import math


class HarmonizationMetric:
    """Base class for harmonization metrics"""
    
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
    
    def compute(self, pred, target, mask):
        """
        Compute metric between prediction and target
        
        Args:
            pred: Predicted image (B, H, W, C) in [0, 1]
            target: Target image (B, H, W, C) in [0, 1]
            mask: Binary mask (B, H, W) in [0, 1]
            
        Returns:
            Metric value (float)
        """
        raise NotImplementedError


class MSE(HarmonizationMetric):
    """Mean Squared Error over entire image"""
    
    def compute(self, pred, target, mask):
        return ((pred - target) ** 2).mean().item()


class fMSE(HarmonizationMetric):
    """Foreground MSE - Only masked region"""
    
    def compute(self, pred, target, mask):
        # Ensure mask has channel dimension
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1)  # (B,H,W) -> (B,H,W,1)
        
        # Compute squared difference only in masked region
        diff = mask * ((pred - target) ** 2)
        
        # Average over pixels and channels
        num_pixels = diff.size(-1) * (mask.sum() + self.epsilon)
        return (diff.sum() / num_pixels).item()


class PSNR(MSE):
    """Peak Signal-to-Noise Ratio over entire image"""
    
    def compute(self, pred, target, mask):
        mse = super().compute(pred, target, mask)
        squared_max = target.max().item() ** 2
        return 10 * math.log10(squared_max / (mse + self.epsilon))


class fPSNR(fMSE):
    """Foreground PSNR - Only masked region (KEY METRIC!)"""
    
    def compute(self, pred, target, mask):
        fmse = super().compute(pred, target, mask)
        squared_max = target.max().item() ** 2
        return 10 * math.log10(squared_max / (fmse + self.epsilon))


class MetricsCalculator:
    """Calculate multiple metrics at once"""
    
    def __init__(self):
        self.metrics = {
            'MSE': MSE(),
            'fMSE': fMSE(),
            'PSNR': PSNR(),
            'fPSNR': fPSNR(),
        }
    
    def compute_all(self, pred, target, mask):
        """
        Compute all metrics
        
        Returns:
            dict: {metric_name: value}
        """
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.compute(pred, target, mask)
        return results
    
    def get_score(self, pred, target, mask):
        """
        Get composite score for ranking models
        
        Higher fPSNR = better harmonization
        """
        return self.metrics['fPSNR'].compute(pred, target, mask)