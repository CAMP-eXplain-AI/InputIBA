import setuptools


setuptools.setup(
    name="iba",
    url="https://github.com/BioroboticsLab/IBA",
    version="0.0.1",
    author="Karl Schulz, Leon Sixt",
    author_email="karl.schulz@tum.de, leon.sixt@fu-berlin.de",
    license='MIT',
    description="Information Bottlenecks for Attribution (iba)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=['test', 'notebooks', 'tools']),
    python_requires='>=3.6',
    keywords=['Deep Learning', 'Attribution', 'XAI'],
)
