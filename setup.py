from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as f:
    required = f.read().splitlines()

setup(
    name="maytracker",
    version="0.1",
    author="Juntae Kim",
    author_email="jtkim@may-i.io",
    description="mAy-I's Tracker Library",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=required,
    python_requires=">=3.10",
)
