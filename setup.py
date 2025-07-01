from setuptools import setup, find_packages

def get_requirements(file_path):
    with open(file_path,'r') as file:
        vals = file.read().splitlines()
        print(vals)
        vals.pop()
        return vals
setup(
    name='mlproj',
    version='0.1.0',
    author='Sarfraz Nawaz',
    packages=find_packages(),
    install_requires=get_requirements('requirments.txt')
)