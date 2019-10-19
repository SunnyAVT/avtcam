from setuptools import setup, find_packages

setup(
    name='avtcam',
    version='0.0.0',
    description='An easy to use Allied Vision camera interface for NVIDIA Jetson',
    long_description=(
          "avtcam is a Python wrapper for application usage. avtcam use the existing module which wrapped Allied Vision's Vimba C API."
          "file included in the Vimba installation to provide a simple Python interface for Allied "
          "Vision cameras. It currently supports only basic functionality provided by Vimba."
      ),
    packages=[
        'avtcam',
    ],
    keywords='python, python3, opencv, cv, machine vision, computer vision, image recognition, vimba, allied vision',
    author='SunnyAVT',
    author_email='lxftm@hotmail.com',
    url='https://github.com/SunnyAVT/avtcam',
    zip_safe=False,
    install_requires=[
      'numpy',
    ]
)

# python3 -m pip install --user --upgrade setuptools wheel twine
# python3 setup.py sdist bdist_wheel
# python3 -m twine upload dist/*
