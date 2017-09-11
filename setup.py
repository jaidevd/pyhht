from setuptools import setup

NAME = "pyhht"

setup(
    name=NAME,
    version='0.1.0',
    author='Jaidev Deshpande',
    author_email='deshpande.jaidev@gmail.com',
    packages=['pyhht'],
    license="New BSD License",
    install_requires=['numpy', 'scipy', 'matplotlib', 'six']
)
