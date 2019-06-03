from setuptools import setup

setup(
    name='pepeiao',
    author='Grady Weyenberg',
    author_email='grady.weyenberg@hawaii.edu',
    packages=['pepeiao'],
    license='MIT',
    install_requires=[
        'h5py',
        'keras',
        'librosa',
        'numpy',
        'pillow',
        'PyWavelets',
        'sounddevice'
        ],
    entry_points={
        'console_scripts':['pepeiao = pepeiao.__main__:_main'],
        'pepeiao_models': ['conv = pepeiao.models:conv_model',
                           'bulbul = pepeiao.models:bulbul',
                           'gru = pepeiao.models:gru_model']
        }
)
