from setuptools import find_packages, setup


setup(
    name="chrysalis",
    version="1.0.0",
    description="Behavioral regression testing for NLP classifiers using metamorphic testing.",
    packages=find_packages(),
    include_package_data=True,
    package_data={"chrysalis": ["registry/mr_registry.yaml", "ci/github_actions_template.yml"]},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "spacy>=3.6.0",
        "nltk>=3.8.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pytest>=7.4.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.10",
)
