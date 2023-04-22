"""Python setup.py for package"""
import io
import os

from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [line.strip() for line in read(path).split("\n") if not line.startswith(('"', "#", "-", "git+"))]


setup(
    name="promptify",
    version=read("promptify", "VERSION"),
    description="Use GPT or other prompt based models to get structured output",
    url="https://github.com/promptslab/Promptify",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="Apache",
    author="monk1337",
    maintainer="The promptslab team with the help of all our contributors",
    packages=find_packages(),
    package_data={"promptify.prompts.text2text": ["*/*.jinja", "*/metadata.json"]},
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.7.0",
    extras_require={"dev": read_requirements("requirements-dev.txt")},
)
