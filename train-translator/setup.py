from io import open

from setuptools import find_packages, setup

console_scripts = []
print("Hello","{0}={1}.app:main".format(
        find_packages("src")[0].replace("_", "-"), find_packages("src")[0]
    ))
console_scripts.append(
    "{0}={1}.app:main".format(
        find_packages("src")[0].replace("_", "-"), find_packages("src")[0]
    )
)
print(console_scripts)
setup(
    entry_points={"console_scripts": console_scripts},
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
print("find_packages:\n",find_packages(where="src"))