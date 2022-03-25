from setuptools import setup

setup(
    name='reggol',
    version='',
    packages=['reggol', 'reggol.logger'],
    url='',
    license='',
    author='Harsha Kokel',
    author_email='harshakokel@gmail.com',
    description='Logger util',
    install_requires=["numpy >= 1.14.2",
                      "GitPython~=3.1.24",
                      "python-dateutil~=2.7.5"
                     ],
)
