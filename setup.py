from setuptools import setup, find_packages

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


reqs = parse_requirements('requirements.txt')


setup(
    name='torchto',
    version='0.1',
    packages=find_packages(include=['torchto', 'torchto.*']),
    install_requires=reqs,)
