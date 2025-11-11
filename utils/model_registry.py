"""
Central registry for all harmonization models
"""

from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class ModelConfig:
    """Configuration for a harmonization model"""
    name: str
    type: str  # 'cdtnet', 'issam', 'harmonizer', 'enhancer', 'pctnet'
    checkpoint: str
    resolution: int
    has_refine: bool = False
    description: str = ""
    speed: str = "medium"  # 'fast', 'medium', 'slow'
    quality: str = "medium"  # 'good', 'better', 'best'


class ModelRegistry:
    """Registry of all available harmonization models"""
    
    _models: Dict[str, ModelConfig] = {
        "CDTNet_iHarmony4_256": ModelConfig(
            name="CDTNet_iHarmony4_256",
            type="cdtnet",
            checkpoint="CDTNet_iHarmony4_256.pth",
            resolution=256,
            has_refine=True,
            description="CDTNet trained on iHarmony4 dataset, balanced quality/speed",
            speed="medium",
            quality="better"
        ),
        "CDTNet_HAdobe5k_2048": ModelConfig(
            name="CDTNet_HAdobe5k_2048",
            type="cdtnet",
            checkpoint="CDTNet_HAdobe5k_2048.pth",
            resolution=2048,
            has_refine=True,
            description="CDTNet for high-resolution images, best quality",
            speed="slow",
            quality="best"
        ),
        "CDTNet_sim_base256": ModelConfig(
            name="CDTNet_sim_base256",
            type="cdtnet",
            checkpoint="CDTNet_sim_base256.pth",
            resolution=256,
            has_refine=False,
            description="Simplified CDTNet, fastest option",
            speed="fast",
            quality="good"
        ),
        "iSSAM_256": ModelConfig(
            name="iSSAM_256",
            type="issam",
            checkpoint="issam256.pth",
            resolution=256,
            has_refine=False,
            description="Lightweight harmonization model",
            speed="fast",
            quality="good"
        ),
        "Harmonizer": ModelConfig(
            name="Harmonizer",
            type="harmonizer",
            checkpoint="harmonizer.pth",
            resolution=256,
            description="Filter-based harmonization (subtle adjustments)",
            speed="fast",
            quality="good"
        ),
        "Enhancer": ModelConfig(
            name="Enhancer",
            type="enhancer",
            checkpoint="enhancer.pth",
            resolution=256,
            description="Filter-based enhancement variant",
            speed="fast",
            quality="good"
        ),
        "PCTNet_CNN": ModelConfig(
            name="PCTNet_CNN",
            type="pctnet",
            checkpoint="PCTNet_CNN.pth",
            resolution=256,
            description="PCTNet (CNN variant) from RakutenTech",
            speed="medium",
            quality="better"
        ),
        "PCTNet_ViT": ModelConfig(
            name="PCTNet_ViT",
            type="pctnet",
            checkpoint="PCTNet_ViT.pth",
            resolution=256,
            description="PCTNet (ViT variant) from RakutenTech",
            speed="medium",
            quality="better"
        ),
    }
    
    @classmethod
    def get_config(cls, model_name: str) -> ModelConfig:
        """Get configuration for a model"""
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")
        return cls._models[model_name]
    
    @classmethod
    def get_all_models(cls) -> List[str]:
        """Get list of all model names"""
        return list(cls._models.keys())
    
    @classmethod
    def get_models_by_type(cls, model_type: str) -> List[str]:
        """Get all models of a specific type"""
        return [name for name, config in cls._models.items() 
                if config.type == model_type]
    
    @classmethod
    def get_fast_models(cls) -> List[str]:
        """Get all fast models"""
        return [name for name, config in cls._models.items() 
                if config.speed == "fast"]
    
    @classmethod
    def get_quality_models(cls) -> List[str]:
        """Get high-quality models"""
        return [name for name, config in cls._models.items() 
                if config.quality in ["better", "best"]]
    
    @classmethod
    def register_model(cls, config: ModelConfig):
        """Register a new model (for future extensibility)"""
        cls._models[config.name] = config