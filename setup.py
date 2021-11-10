from setuptools import setup

setup(
        name="GAGL",
        version="0.0.1",
        author="Umberto Gagliardini",
        author_email="u.gagliardini@studenti.unina.it",
        packages=["gagl"],
        package_dir={"gagl":"gagl"},
        url="https://github.com/umbertogagl97/LivDetGagl/",
        install_requires=[  "numpy >= 1.8" ]
)
