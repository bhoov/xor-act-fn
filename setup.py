from setuptools import setup, find_packages

requires = [] # Let conda handle requires

setup(
    name="rbfxor",
    description="Piecewise RBF activation function for NN",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    author="Ben Hoover, IBM Research AI",
    include_package_data=True,
    install_requires=requires
)
