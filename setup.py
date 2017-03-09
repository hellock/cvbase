from setuptools import find_packages, setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='cvbase',
    version='0.0.1',
    description='Utils for computer vision research',
    long_description=readme(),
    keywords='computer vision',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities',
    ],
    url='https://github.com/hellock/cvbase',
    author='Kai Chen',
    author_email='chenkaidev@gmail.com',
    license='MIT',
    install_requires=[
        'numpy>=1.11.0',
        'six>=1.10.0'
    ],
    zip_safe=False
)  # yapf: disable
