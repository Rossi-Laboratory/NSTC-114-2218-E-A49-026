from setuptools import setup, find_packages

setup(
    name="vla-sim-env",
    version="0.1.0",
    description="High-fidelity simulation environment for VLA and robotics (Omniverse + Genesis)",
    packages=find_packages(exclude=("tests", "docs", "examples", "scripts", "configs")),
    python_requires=">=3.9",
    install_requires=[
        "pyyaml>=6.0.1",
        "numpy>=1.26.0",
        "rich>=13.7.0",
        "matplotlib>=3.8.0",
        "networkx>=3.2.1",
    ],
)
