from setuptools import setup

setup(
    name='deepnet',
    version='1.0',
    packages=['deepclassify'],
    url='',
    license='',
    author='Adam',
    author_email='',
    description='demo',
    install_requires=['torch', 'torch-vision'],
    package_data={'data':['*.txt']}
)
