from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['cfs_interface'],
    package_dir={'': 'src'}
)
setup(**d)