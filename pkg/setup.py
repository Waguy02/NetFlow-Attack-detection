from setuptools import find_packages, setup

requirements = """
pandas
numpy
"""

setup(
    name="network_ad",
    author="NAME_OF_YOUR_GROUP",
    description="Detect anomalies in IP network using Machine Learning",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='==3.10.12',
    include_package_data=True,
    scripts=[],
    zip_safe=False,
)