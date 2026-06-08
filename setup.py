from setuptools import setup, find_packages

setup(
    name="tsr",
    version="0.1.0",
    description="Topological Self-Regulation: Neural networks that modify their own structure during training",
    author="TSR Research",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchdiffeq>=0.2.3",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
        "fvcore>=0.1.5",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
)
