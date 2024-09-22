from setuptools import setup

setup(name='simulator',
      version='0.0.0',
      install_requires=['gymnasium',
                        'gymnasium-robotics',
                        'stable-baselines3[extra]',
                        'ipykernel',
                        'ipywidgets',
                        'numpy']
)