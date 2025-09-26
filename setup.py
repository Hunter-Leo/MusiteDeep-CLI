from setuptools import setup, find_packages

setup(
    name='musitedeep',
    version='1.0.0',
    description='MusiteDeep PTM prediction command line tool',
    author='MusiteDeep Team',
    py_modules=['cli'],
    install_requires=[
        'click',
        'numpy',
        'scipy',
        'scikit-learn',
        'pillow',
        'h5py',
        'pandas',
        'keras==2.2.4',
        'tensorflow==1.12.0'
    ],
    entry_points={
        'console_scripts': [
            'musitedeep=cli:predict',
        ],
    },
    python_requires='>=3.5',
)
