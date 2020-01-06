from setuptools import setup, find_packages

requires = [] # Let conda handle requires

setup(
    name="rbf-actfunc",
    description="Piecewise RBF activation function for NN",
    packages=find_packages(),
    author="Ben Hoover, IBM Research AI",
    include_package_data=True,
    install_requires=requires
)
