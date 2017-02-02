from setuptools import setup
setup(
  name = 'PyAPL',
  packages = ['PyAPL'],
    version='0.2.1',
  description = 'A Python interpreter for the APL programming language',
  author = 'Matt Torrence',
  author_email = 'matt@torrencefamily.net',
  url = 'https://github.com/Torrencem/PyAPL',
    download_url='https://github.com/Torrencem/PyAPL/tarball/0.2.1',
  keywords = ['interpreter', 'APL'],
  classifiers = [],
  install_requires=[
          'ply',
          'numpy'
      ]
)
