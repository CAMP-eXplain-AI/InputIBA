import os.path as osp
import pkg_resources
from setuptools import find_packages, setup

import pathlib


def fetch_requirements():
    with pathlib.Path("requirements.txt").open() as requirements_txt:
        install_requires = [
            str(requirement) for requirement in
            pkg_resources.parse_requirements(requirements_txt)
        ]
    try:
        _ = pkg_resources.get_distribution('mmcv-full')
        # mmcv-full is already installed,
        # so remove mmcv from the requirements list
        install_requires = [x for x in install_requires if 'mmcv' not in x]
    except pkg_resources.DistributionNotFound:
        pass

    return install_requires


def get_version():
    init_py_path = osp.join(
        osp.abspath(osp.dirname(__file__)), "input_iba", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [
        l.strip() for l in init_py  # noqa: E741
        if l.startswith("__version__")
    ][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


packages = find_packages(exclude=['tests', 'tools'])

setup(
    name="input_iba",
    url="",
    version=get_version(),
    author="Anonymous",
    author_email="anonymous@anonymous.com",
    license='MIT',
    description=
    "Fine-grained neural network explanation by identifying input features with predictive information.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=fetch_requirements(),
    packages=packages,
    extras_require={'dev': ['pytest', 'sphinx', 'flake8', 'yapf']},
    python_requires='>=3.7',
    keywords=['Deep Learning', 'Attribution', 'XAI'],
)
