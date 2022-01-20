import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="benchENAS",
    packages=setuptools.find_packages(),
    include_package_data=True,
    version="1.0.1",
    author="xiaoyang",
    author_email="yangx9810@163.com",
    description="A method of benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'paramiko',  'numpy',  'multiprocess',
        'torch==1.6.0', 'torchvision==0.7.0',
        'redis'
    ],
)