import os
from setuptools import find_packages, setup
import versioneer


def resolve_requirements(file):
    if not os.path.isfile(file):
        file = os.path.join(os.path.dirname(__file__), "requirements", file)
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


def unify_requirements(base_requirements: list, *additional_requirement_lists):
    for reqs in additional_requirement_lists:
        for req in reqs:
            if req not in base_requirements:
                base_requirements.append(req)

    return base_requirements


def parse_all_requirements(backend_requirement_dict: dict):
    backend_requirements = {"full": []}

    # parse all requirements
    for backend_name, requirement_file in backend_requirement_dict.items():
        _reqs = resolve_requirements(requirement_file)
        backend_requirements[backend_name] = _reqs

        # add all requirements to full if not already part of it
        backend_requirements["full"] = unify_requirements(
            backend_requirements["full"], _reqs)

    # for each backend: check if requirement is already in base requirements
    for backend_name, reqs in backend_requirements.items():
        if backend_name == "base":
            continue

        for _req in reqs:
            if _req in backend_requirements["base"]:
                reqs.pop(reqs.index(_req))

        backend_requirements[backend_name] = reqs

    return backend_requirements


requirement_files = {
    "base": "base.txt",
    "sklearn": "base.txt",  # no extra requirements necessary
    "torch": "torch.txt",
    "torchscript": "torch.txt",
    "tensorflow": "tensorflow.txt",
    "tensorflow_eager": "tensorflow.txt",
    "chainer": "chainer.txt"
}


requirement_dict = parse_all_requirements(requirement_files)

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
    install_requires=requirement_dict.pop("base"),
    tests_require=["coverage"],
    python_requires=">=3.5",
    extras_require=requirement_dict
)
