from setuptools import setup

REQUIREMENTS = [
    # 'nltk',
    # 'pandas',
    # 'tqdm',
    # 'psutil',
    # 'sklearn',
    # 'fre @ git+https://github.com/ljdyer/feature-restoration-evaluator.git'
]

setup(
    name='bilstm_char_feature_restorer',
    version='0.1',
    description="""\
Train character-level BiLSTM models for restoration of features such as spaces, \
punctuation, and capitalization to unformatted texts\
""",
    author='Laurence Dyer',
    author_email='ljdyer@gmail.com',
    url='https://github.com/ljdyer/BiLSTM-Char-Feature-Restorer',
    packages=['bilstm_char_feature_restorer'],
    install_requires=REQUIREMENTS
)
