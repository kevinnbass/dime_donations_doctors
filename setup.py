#!/usr/bin/env python3
"""Setup script for dime_donations_doctors package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="dime_donations_doctors",
    version="1.0.0",
    author="Kevin Bass",
    description="Analysis of physician political donations from DIME database (1980-2024)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevinnbass/dime_donations_doctors",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    entry_points={
        "console_scripts": [
            "replicate-figure=replicate_figure:main",
        ],
    },
)
