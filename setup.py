import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gsuppose",
    version="0.1",
    author="Laboratorio de Fotónica, Universidad de Buenos Aires",
    author_email="alacapmesure@fi.uba.ar",
    description="SUPPOSe deconvolution algorithm by means of gradient descent optimization methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/labofotonica/gsuppose",
    packages=setuptools.find_packages(where="gsuppose"),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: GPU :: NVIDIA CUDA',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Typing :: Typed',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'matplotlib', 'tifffile'],
    package_dir={"":"gsuppose"},
    include_package_data=True
)
