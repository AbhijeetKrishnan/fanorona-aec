from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gym-fanorona",
    version="0.0.1",
    description="An OpenAI Gym implementation of the Fanorona board game.",
    url="https://github.com/AbhijeetKrishnan/gym-fanorona",
    author="Abhijeet Krishnan",
    author_email="abhijeet.krishnan@gmail.com",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=["gym"],
    tests_require=["pytest"],
)
