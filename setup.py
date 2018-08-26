from setuptools import setup, find_packages

requires = [
    'six',
    'requests>=2.17.3',
    'uuid',
    'gunicorn',
    'Flask',
    'numpy',
    'pandas',
    'python-dateutil',
    'querystring_parser',
]

setup(
    name='flask_todo',
    version='0.0',
    description='A Simple Machine Learning Workflow',
    author='F2',
    author_email='<Your actual e-mail address here>',
    keywords='web flask',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requires
)
