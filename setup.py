import os
import re
from setuptools import find_packages, setup
import versioneer


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


requirements = resolve_requirements(os.path.join(os.path.dirname(__file__),
                                                 'requirements.txt'))

requirements_extra_full = []

requirements_extra_torch = resolve_requirements(os.path.join(
    os.path.dirname(__file__), 'requirements_extra_torch.txt'))
requirements_extra_full += requirements_extra_torch

requirements_extra_tf = resolve_requirements(os.path.join(
    os.path.dirname(__file__), 'requirements_extra_tf.txt'))
requirements_extra_full += requirements_extra_tf

readme = read_file(os.path.join(os.path.dirname(__file__), "README.md"))

setup(
    name='delira',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    url='https://github.com/delira-dev/delira/',
    test_suite="unittest",
    long_description=readme,
    long_description_content_type='text/markdown',
    maintainer="Justus Schock",
    maintainer_email="justus.schock@rwth-aachen.de",
    license='BSD-2',
    install_requires=requirements,
    tests_require=["coverage"],
    python_requires=">=3.5",
    extras_require={
        "full": requirements_extra_full,
        "torch": requirements_extra_torch,
        "tensorflow": requirements_extra_tf
    }
)
