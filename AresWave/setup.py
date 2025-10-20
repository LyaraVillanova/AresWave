from setuptools import setup, find_packages

setup(
    name='areswave',
    version='1.0.0',
    description='AresWave: source parameter estimation of marsquakes using a modified DSMpy',
    author='Lyara S. Villanova',
    url='https://github.com/LyaraVillanova/AresWave',
    packages=find_packages(include=['areswave', 'areswave.*']),
    install_requires=[
        'numpy','scipy','pandas','matplotlib','obspy','corner','pyswarms','mpi4py','tqdm'
    ],
    python_requires='>=3.8',
)
