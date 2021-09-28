"""Installation script for basilic - development dependencies in requirements_dev.txt"""

from setuptools import setup

requirements = [
    "numpy",
    "spikeinterface",
    "sklearn",
]

# Usage: pip install -e .[dev]
extra_requirements = {
    "dev": [
        "ipykernel",
        "ipython",
        "jupyterlab",
    ]
}

setup(
    author="Jean de Montigny",
    description="Bayesian Spike Localisation",
    extras_require=extra_requirements,
    install_requires=requirements,
    license="MIT",
    name="basilic",
    packages=["basilic"],
    url="https://github.com/JeandeMontigny/basilic.git",
    version="0.1.0",
)
