from setuptools import setup, find_packages
from setuptools.command.install import install
import shutil
import os

# class CustomInstallCommand(install):
#     def run(self):
#         install.run(self)
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         key_file = os.path.join(current_dir, 'IBM.key')
#         target_dir = os.path.join(self.install_lib, 'qublp/solvers/cloud_execute')
#         if os.path.isfile(key_file):
#             shutil.copy(key_file, target_dir)

def _process_requirements():
    packages = open('requirements.txt').read().strip().split('\n')
    requires = []
    for pkg in packages:
        if pkg.startswith('git+ssh'):
            return_code = os.system('pip install {}'.format(pkg))
            assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)
        else:
            requires.append(pkg)
    return requires

setup(
    name='qto',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[],
    # install_requires=_process_requirements(),
    # classifiers=[
    #     'Programming Language :: Python :: 3',
    #     'License :: OSI Approved :: MIT License',
    #     'Operating System :: OS Independent',
    # ],
    # package_data={
    #     'quBLP.solvers.cloud_execute': ['IBM.key'],
    # },
)
