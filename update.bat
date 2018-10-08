REM set home to locate PyPI login information

set HOME=c:\S\TELOS\Python

python setup.py bdist_egg upload
python setup.py sdist upload
REM python setup.py bdist_wininst --target-version=3.6 upload


REM python setup.py sdist
