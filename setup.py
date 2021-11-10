from setuptools import setup

setup(
        name="GAGL",
        version="0.0.1",
        author="Umberto Gagliardini",
        author_email="u.gagliardini@studenti.unina.it",
        packages=["gagl"],
        package_dir={"gagl":"gagl"},
        url="http://github.com/cehorn/GLRM/",
        install_requires=[  "numpy >= 1.8" ]
)
