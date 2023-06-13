
from setuptools import setup, find_packages, find_namespace_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="nomelt",  
    version="2023.1",  # Required
    long_description=long_description, 
    long_description_content_type="text/markdown", 
    url="https://github.com/BeckResearchLab/nomelt",
    author="Evan Komp, Humood Alanzi, David Beck", 
    author_email="evankomp@uw.com",
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="nlp, proteins",
    packages=find_namespace_packages(),
    python_requires=">=3.7, <4",
    extras_require={ 
        "test": ["coverage", 'pytest'],
    },

    package_data={},
    data_files=None, 
    entry_points={},
)
