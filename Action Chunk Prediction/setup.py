from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements.txt, ignoring comments and blank lines."""
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
    reqs = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs


setup(
    name="action-chunk-prediction",
    version="0.1.0",
    description="A research framework for Action Chunk Prediction (ACP) to improve multi-step robot behavior learning using chunked action sequences.",
    author="Yuan-Fu Yang",
    author_email="yuanfuyang@nycu.edu.tw",
    url="https://github.com/your-org/Action-Chunk-Prediction",
    license="Apache-2.0",
    packages=find_packages(include=["acp*", "scripts*", "configs*", "docs*"]),
    include_package_data=True,
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "acp-train=acp.training.train:main",
            "acp-decode=acp.inference.decode:main",
            "acp-export=acp.inference.export:main",
        ],
    },
    keywords="robotics VLA MoE action-chunk prediction manipulation",
)
