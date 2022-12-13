@ECHO OFF

REM to this by hand
REM call makerst

REM pushd stores the current directory and sets the cwd to its argument
REM percent i is the i-th argument
REM percent tilde i expands the argument and removes any quotes
REM See https://stackoverflow.com/questions/5034076/what-does-dp0-mean-and-how-does-it-work

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=python -msphinx
)
set SOURCEDIR=.
set BUILDDIR=_build
set SPHINXPROJ=aggregate

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The Sphinx module was not found. Make sure you have Sphinx installed,
	echo.then set the SPHINXBUILD environment variable to point to the full
	echo.path of the 'sphinx-build' executable. Alternatively you may add the
	echo.Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

REM append -a -E for a more ground up rebuild

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end


REM xcopy /S .\_build\singlehtml\*.* \s\telos\python\aggregate\


REM get back to where you started
popd
