from setuptools import setup, find_packages

setup(
    name="action-chunk-prediction",
    version="0.1.0",
    description="Action Chunk Prediction (ACP) with FAST & VQ; integrates VLA-MoE-Manipulation",
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "acp-train=acp.training.train:main",
            "acp-infer=acp.inference.run_inference:main",
        ],
    },
)
