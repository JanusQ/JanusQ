from setuptools import find_packages, setup

with open("README.md", "r") as f:
  long_description = f.read()

with open('requirements.txt') as f:
    requirements = [line.replace('\n', '') for line in f.readlines()]

setup(name='janusq',    # 包名
      version='0.1.0',        # 版本号
      description='janusq: a full-stack framework.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='ZJU-CCNT',
      author_email='',
      url='',
      install_requires=requirements,	# 依赖包会同时被安装
      license='MIT',
      packages=find_packages(),
      python_requires='>=3.6',
      include_package_data=True,
      )