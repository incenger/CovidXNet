from setuptools import setup, find_packages

setup(
    name='CovidX Net',
    version='1.0.0',
    include_package_data=True,
    packages=find_packages(exclude=["data", "scripts"]),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",
        "scikit-learn",
        "matplotlib",
        "tensorboard",
        "pytorch-lightning",
        "efficientnet_pytorch",
    ],
    python_requires=">=3.6",
)
