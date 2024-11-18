from setuptools import setup, find_packages

setup(
    name="cat_finder",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    url="https://github.com/leareuter/catfinder",
    license="LGPLv3+",
    license_files = "LICENSE.md",
    author="Lea Reuter et al.",
    author_email="lea.reuter@kit.edu",
    description="Belle II CAT Finder",
)
