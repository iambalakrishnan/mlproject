from setuptools import setup,find_packages
from typing import List

#Declaring variables for setup functions

REPO_NAME = "mlproject"
AUTHOR_USER_NAME = "iambalakrishnan"
SRC_REPO = "src"
REQUIREMENT_FILE_NAME="requirements.txt"

HYPHEN_E_DOT="-e ."

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


def get_requirements_list()->List[str]:
    """
    Description: This function will return list of requirement
    mention in requirements.txt file
    return: This function is going to return a list which contain name
    of libraries mentioned in requirements.txt file
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list: List[str] = requirement_file.readlines()
        requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]
        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)
        return requirement_list

setup(
    name=SRC_REPO,
    version="0.0.2",
    author=AUTHOR_USER_NAME,
    description="An end-to-end mlproject",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="imbkrishnaa@gmail.com",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.8",
    install_requires=get_requirements_list()
)