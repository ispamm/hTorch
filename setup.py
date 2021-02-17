import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qtorch-giorgiozannini",
    version="0.0.1",
    author="Giorgio Zannini",
    author_email="giorgiozannini97@gmail.com",
    description="PyTorch extension to support quaternion-valued tensors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giorgiozannini/QNN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)