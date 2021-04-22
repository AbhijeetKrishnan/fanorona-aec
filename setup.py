from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="fanorona-aec",
    version="1.0.0",
    description="A PettingZoo AECEnv implementation of the Fanorona board game.",
    url="https://github.com/AbhijeetKrishnan/fanorona-aec",
    author="Abhijeet Krishnan",
    author_email="abhijeet.krishnan@gmail.com",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["fanorona_aec"],
    python_requires=">=3.8",
    install_requires=["pettingzoo"],
    tests_require=["pytest"],
)
