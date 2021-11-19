from setuptools import setup

setup(
    name='play-to-grade',
    description="Environments for doing RL on student programming assignment grading",
    author="windfeller",
    packages=["bounce", "car"],
    install_requires=[
        "torch",
        "sklearn",
        "numpy",
        "tqdm",
        "pygame",
    ]
)