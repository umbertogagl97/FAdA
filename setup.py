from setuptools import setup

setup(
        name="LivDet_ART",
        version="0.0.1",
        description="Custom ART for Fingerprint Liveness Detection",
        packages=["ldart"],
        author="Umberto Gagliardini",
        author_email="u.gagliardini@studenti.unina.it",
        url="https://github.com/umbertogagl97/LivDet_art/",
        install_requires=[  "numpy >= 1.8" ],
        include_package_data=True,
)
