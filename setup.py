from setuptools import setup, find_packages

setup(
    name="vision",
    version="0.1",
    license="GNU GPLv3.0",
    packages=find_packages(),
    install_requires=['spynnaker', "numpy", ],
    classifiers = [
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",

        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
    ]
)
