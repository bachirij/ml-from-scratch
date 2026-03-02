from setuptools import setup, find_packages

setup(
    name="ml-from-scratch",
    version="0.1.0",
    description="Machine Learning algorithms implemented from scratch",
    author="Jihane Bachiri",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
    ],
)
