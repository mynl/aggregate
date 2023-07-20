REM start /B xelatex DecL_Cheat_Sheet.tex         > dmd.log
REM start /B xelatex Aggregate_Cheat_Sheet.tex    > dma.log
REM start /B xelatex Distortion_Cheat_Sheet.tex   > dmdi.log
REM start /B xelatex Portfolio_Cheat_Sheet.tex    > dmdp.log
REM start /B xelatex Severity_Cheat_Sheet.tex     > dmd.sog
REM start /B xelatex Underwriter_Cheat_Sheet.tex  > dmu.log

start /B lualatex -interaction=batchmode DecL_Cheat_Sheet.tex
start /B lualatex -interaction=batchmode Aggregate_Cheat_Sheet.tex
start /B lualatex -interaction=batchmode Distortion_Cheat_Sheet.tex
start /B lualatex -interaction=batchmode Portfolio_Cheat_Sheet.tex
start /B lualatex -interaction=batchmode Severity_Cheat_Sheet.tex
start /B /WAIT lualatex -interaction=batchmode Underwriter_Cheat_Sheet.tex

