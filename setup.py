from setuptools import setup, find_packages

setup(
    name='CovidX Net',
    version='1.0.0',
    include_package_data=True,
    packages=find_packages(exclude=["data", "scripts"])
    install_requires=[],
    python_requires=">=3.6",
)
