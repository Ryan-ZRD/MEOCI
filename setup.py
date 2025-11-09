"""
setup.py
------------------------------------------------------------
Installation script for the MEOCI framework.

Usage:
    pip install -e .

This enables editable installation for module imports such as:
    from core.agent import agent_adp_d3qn
    from visualization.ablation import plot_ablation_convergence
------------------------------------------------------------
"""

from setuptools import setup, find_packages

# ------------------------------------------------------------
# Project metadata
# ------------------------------------------------------------
setup(
    name="MEOCI",
    version="1.0.0",
    author="Intelligent Edge Computing Research Group",
    author_email="research@meoci.org",
    description=(
        "MEOCI: Model Partitioning and Early-Exit Point Selection "
        "Joint Optimization for Collaborative Inference in Vehicular Edge Computing."
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YourUsername/MEOCI",
    license="MIT",

    # --------------------------------------------------------
    # Package discovery
    # --------------------------------------------------------
    packages=find_packages(
        exclude=[
            "tests*",
            "docs*",
            "examples*",
            "*.egg-info",
            "build",
            "dist"
        ]
    ),

    include_package_data=True,

    # --------------------------------------------------------
    # Dependencies (ensure reproducibility)
    # --------------------------------------------------------
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "matplotlib>=3.8.0",
        "pandas>=2.1.0",
        "pyyaml>=6.0",
        "scipy>=1.11.0",
        "tqdm>=4.66.0",
        "tensorboard>=2.14.0",
        "seaborn>=0.13.0",
        "psutil>=5.9.0",
        "flask>=3.0.0",
        "prometheus-client>=0.17.0",
        "influxdb-client>=1.41.0",
    ],

    # --------------------------------------------------------
    # Optional dependencies (for visualization)
    # --------------------------------------------------------
    extras_require={
        "dev": ["pytest>=8.0", "black>=24.0", "flake8>=6.1"],
        "visual": ["plotly>=5.18.0", "notebook>=7.0.0"]
    },

    # --------------------------------------------------------
    # Entry points for command-line interface
    # --------------------------------------------------------
    entry_points={
        "console_scripts": [
            "meoci=run:main",
        ]
    },

    # --------------------------------------------------------
    # Python compatibility
    # --------------------------------------------------------
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
