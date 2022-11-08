@echo off
REM set home to locate PyPI login information

rem set HOME=c:\S\TELOS\Python


echo building
python setup.py bdist_egg
python setup.py sdist

REM python setup.py bdist_wininst --target-version=3.6 upload
REM python setup.py sdist

REM upload
REM echo uname is mildenhall pw is for pypi

if %1% == u (
    echo uploading
    twine -u mildenhall -p %PPPW% upload dist/*
) else (
    echo not uploading
)
