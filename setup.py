import setuptools


setuptools.setup(
    name="iba",
    url="",
    version="0.0.1",
    author="Anonymous",
    author_email="anonymous@anonymous.com",
    license='MIT',
    description="Information Bottlenecks for Attribution (iba)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=['tests', 'notebooks', 'tools']),
    python_requires='>=3.7',
    keywords=['Deep Learning', 'Attribution', 'XAI'],
)
