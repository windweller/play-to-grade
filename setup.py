from setuptools import setup

setup(
    name='play-to-grade',
    description="Environments for training agents to interactively evaluate student programming submissions",
    author="windweller",
    packages=["bounce", "car"],
    install_requires=[
        "torch",
        "sklearn",
        "numpy",
        "tqdm",
        "pygame",
    ]
)
