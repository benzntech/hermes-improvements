from setuptools import setup, find_packages

setup(
    name="hermes-improvements",
    version="2.0.0",
    description="Architectural enhancement suite for Hermes Agent: vector memory, adaptive soul, and workflow classification.",
    author="Community Contributor",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
