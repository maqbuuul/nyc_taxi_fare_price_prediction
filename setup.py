from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    This function returns the list of requirements from the requirements.txt file
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="nyc_taxi_fare_prediction",
    version="0.1",
    author="Abdiwahid Ali",
    author_email="abdiwahid.ali@example.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
