from pathlib import Path

import setuptools
from numpy.distutils.core import setup, Extension


ROOT = Path(__file__).parent.resolve()


def read_requirements():
    requirements_file = ROOT / "requirements.txt"
    return [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


# AresWave package
areswave_packages = setuptools.find_packages(
    include=["areswave", "areswave.*"]
)

# Modified DSMpy bundled with AresWave.
# The actual Python package is located at dsmpy/dsmpy/.
dsmpy_subpackages = setuptools.find_packages(
    where="dsmpy/dsmpy"
)
dsmpy_packages = ["dsmpy"] + [
    f"dsmpy.{package}" for package in dsmpy_subpackages
]


lib_tish = Extension(
    name="dsmpy.flib.tish",
    sources=[
        "dsmpy/dsmpy/src_f90/tish/parameters.f90",
        "dsmpy/dsmpy/src_f90/tish/tish.f90",
        "dsmpy/dsmpy/src_f90/tish/others.f90",
        "dsmpy/dsmpy/src_f90/tish/calmat.f90",
        "dsmpy/dsmpy/src_f90/tish/trialf.f90",
        "dsmpy/dsmpy/src_f90/tish/dclisb.f90",
    ],
    extra_f90_compile_args=["-Ofast"],
    extra_f77_compile_args=["-Ofast"],
)


lib_tipsv = Extension(
    name="dsmpy.flib.tipsv",
    sources=[
        "dsmpy/dsmpy/src_f90/tipsv/parameters.f90",
        "dsmpy/dsmpy/src_f90/tipsv/tipsv.f90",
        "dsmpy/dsmpy/src_f90/tipsv/others.f90",
        "dsmpy/dsmpy/src_f90/tipsv/calmat.f90",
        "dsmpy/dsmpy/src_f90/tipsv/trialf.f90",
        "dsmpy/dsmpy/src_f90/tipsv/dcsymbdl.f90",
        "dsmpy/dsmpy/src_f90/tipsv/glu2.f90",
        "dsmpy/dsmpy/src_f90/tipsv/rk3.f90",
    ],
    extra_f90_compile_args=["-Ofast"],
    extra_f77_compile_args=["-Ofast"],
)


with open(ROOT / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="AresWave",
    version="1.0",
    author="Lyara Villanova",
    author_email="lyaravillanova@yahoo.com",
    license="MIT",
    description=(
        "Waveform-fitting framework for marsquake "
        "source-parameter estimation"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LyaraVillanova/AresWave",

    packages=areswave_packages + dsmpy_packages,

    # dsmpy is physically stored one level deeper because the modified
    # upstream DSMpy repository is bundled inside AresWave.
    package_dir={
        "dsmpy": "dsmpy/dsmpy",
    },

    package_data={
        "dsmpy.resources": ["scardec.pkl"],
    },
    include_package_data=True,

    install_requires=read_requirements(),

    ext_modules=[
        lib_tish,
        lib_tipsv,
    ],

    python_requires=">=3.9,<3.12",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
