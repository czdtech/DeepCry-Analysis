from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="audioset_utils",
    version="0.1.0",
    packages=find_packages(),
    author="Bilal Ahmed",
    author_email="bilalhsp@gmail.com",
    url="https://github.com/bilalhsp/audioset-utils",
    description="Utility for downloading subsets of audioset.",
    long_description=long_description,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'csc', 'pandas', 'yt_dlp', 'pydub', 'datasets'
    ],
    python_required='>=3.7',
)


