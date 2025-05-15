from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deepfake-detection-project",
    version="0.1.0",
    author="Yasin Seyhun",
    author_email="seyhunyasin@gmail.com",
    description="A deep learning project for deepfake detection using StyleGAN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YasinSeyhun/deepfake-detection-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        line.strip()
        for line in open("requirements/base.txt").readlines()
        if not line.startswith("#")
    ],
) 