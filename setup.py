from setuptools import find_packages, setup

with open("README.md", "r") as f:
  long_description = f.read()

with open('requirements.txt') as f:
    requirements = [line.replace('\n', '') for line in f.readlines()]

setup(name='janusq',    # 包名
      version='0.3.0',        # 版本号
      description='A Software Framework for Analyzing, Optimizing, Verifying and Implementing Quantum Circuit.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='ZJU-ACES',
      author_email='',
      url='https://github.com/JanusQ/JanusQ',
      install_requires=requirements,	# 依赖包会同时被安装
      license='MIT',
      packages=find_packages(),
      python_requires='>=3.10',
      include_package_data=True,
      )
