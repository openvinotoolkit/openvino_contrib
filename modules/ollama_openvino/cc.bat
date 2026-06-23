@echo off
setlocal enabledelayedexpansion
set "args="
:loop
if "%~1" == "" goto :run
set "arg=%~1"
if "!arg!" == "-Werror" goto :next
if "!arg!" == "-Wall" goto :next
if "!arg!" == "-m64" goto :next
if "!arg!" == "-mthreads" goto :next
if "!arg!" == "-fno-stack-protector" goto :next
if "!arg!" == "-fmessage-length=0" goto :next
if "!arg!" == "-g" (
    set "args=!args! /Zi"
    goto :next
)
if "!arg!" == "-O2" (
    set "args=!args! /O2"
    goto :next
)
if "!arg!" == "-o" (
    set "args=!args! /Fo%~2"
    shift
    goto :next
)
if "!arg!" == "-c" (
    set "args=!args! /c"
    goto :next
)
set "args=!args! %1"
:next
shift
goto :loop
:run
cl.exe !args!
