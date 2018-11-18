import os
import setuptools 

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setuptools.setup(
    name = "ukuxhumana",
    version = "0.0.1",
    author = "Laura martinus,Jade Abbott",
    author_email = "jabbott@retrorabbit.co.za",
    description = ("Neural Machine Translation for African Languages"),
    license = "gpl-3.0",
    keywords = "neural machine translation african languages",
    url = "https://github.com/LauraMartinus/ukuxhumana",
    packages=setuptools.find_packages(),
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Machine Translation",
        "License ::  GPL 3.0",
    ],
    install_requires=[
        'tensor2tensor'
    ],
)