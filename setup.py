from setuptools import setup, find_packages
import os

# Install_requires
thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = [] # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()


setup(
    name='aco_mixed',
    version='0.1.0',
    author='RheoTabs',
    author_email='theo.rabut@gmail.fr',
    packages=find_packages(),
    license='LICENSE.txt',
    description='TODO',
    long_description=open('README.md').read(),
    install_requires=install_requires,
)
