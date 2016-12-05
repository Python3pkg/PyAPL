from distutils.core import setup
setup(
  name = 'PyAPL',
  packages = ['PyAPL'],
  version = '0.1.2',
  description = 'A Python interpreter for the APL programming language',
  author = 'Matt Torrence',
  author_email = 'matt@torrencefamily.net',
  url = 'https://github.com/Torrencem/PyAPL',
  download_url = 'https://github.com/Torrencem/PyAPL/tarball/0.1.2',
  keywords = ['interpreter', 'APL'],
  classifiers = [],
  install_requires=[
          'ply'
      ]
)