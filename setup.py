from setuptools import setup

setup(
        name="FAdA",
        version="0.1.0",
        description="Fingerprint ADversarial Attacks",
        packages=["fada"],
        package_dir={"fada":"fada"},
        author="Umberto Gagliardini",
        author_email="umberto.salv@gmail.com",
        url="https://github.com/umbertogagl97/FAda/",
        install_requires=[  "numpy >= 1.8" ],
        include_package_data=True,
)
