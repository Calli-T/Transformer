title Infer Diff-SVC
SETLOCAL ENABLEDELAYEDEXPANSION

REM ================================
REM root 는 anaconda3의 설치 경로를 입력해줍니다
REM ================================

set root=C:\ProgramData\anaconda3
set dpath=I:\_Diff-svc

REM ================================
REM dpath 는 Diff-svc의 설치 경로를 입력해줍니다
REM ================================


set cpath=%dpath%\checkpoints\
set "ccnt=0"
set "acnt=0"
sef df0=
set df1=0102_xiaoma_pe
set df2=0109_hifigan_bigpopcs_hop128
set df3=hubert
set df4=nsf_hifigan
echo off
cls
cd /d %dpath%
for /f "tokens=*" %%d in ('dir %cpath% /B /a:d') DO (
if %df1% == %%d ( 
REM echo df1 : %%d 
) else (
if %df2% == %%d ( 
REM echo df2 : %%d 
) else ( 
if %df3% == %%d ( 
REM echo df3 : %%d
) else (
if %df4% == %%d (
REM echo df4 : %%d
) else (
REM echo %%d
set df[!ccnt!]=%%d
set /a ccnt+=1
)))))
:arrayLoop
if defined df[%acnt%] (
    set /a "acnt+=1"
    GOTO :arrayLoop
)
if "%ccnt%" GTR "1" ( set /a "acnt-=1" )
:selectLoop
cls
if %ccnt% == 0 ( goto :notrain )
if %ccnt% == 1 (
    set df0=%df[0]%
    goto :Cok
) else (
for /l %%n in (0,1,!acnt!) do (
    echo %%n : !df[%%n]!
)
)
REM echo %acnt%
echo.
set /p UST= 추론 할 모델명을 선택해주세요. (숫자만 입력) : 
for /L %%a in (0,1,!acnt!) do (
    if "%UST%" == "%%a" (
        set df0=!df[%%a]!
        goto :Cok
    )
)
REM echo f : %UST%
goto :selectLoop




:notrain
endlocal
rundll32 user32.dll,MessageBeep
echo 학습된 CKPT 파일이 checkpoints 폴더에 존재하지 않습니다
pause
exit


:Cok
REM echo %df0%
cls
call %root%\Scripts\activate.bat %root%
call cd /d %dpath%
call conda activate diff-svc
call set PYTHONPATH=.
call set CUDA_VISIBLE_DEVICES=0
call python infer_for_bat.py "%df0%"
endlocal
rundll32 user32.dll,MessageBeep
exit