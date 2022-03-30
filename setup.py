"""
Setup script for pysatellite using PBR package.
(Use 'setup.cfg' to update package details and PBR will pick them up automatically)
"""

from setuptools import setup

# setup(setup_requires=["pbr"],pbr=True)
setup(name='rl-model',
      maintainer='Benedict Oakes',
      maintainer_email='sgboakes@liverpool.ac.uk',
      url='https://github.com/sgboakes/rl-model',
      install_requires=[
          'numpy', 'tf-agents[reverb]', 'reverb', 'pysatellite', 'matplotlib', 'tensorflow', 'pandas', 'filterpy'
      ],
      pbr=True,
      )
