"""Setup script for MemoryPalAI — Multimodal Emotion Recognition with XAI."""

from setuptools import setup, find_packages

setup(
    name="mmer-xai",
    version="0.1.0",
    description="Multimodal Emotion Recognition with Explainable AI and Natural Language Generation",
    author="Abhi",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.36.0",
        "timm>=0.9.12",
        "shap>=0.43.0",
        "pytorch-grad-cam>=1.5.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
    ],
)
