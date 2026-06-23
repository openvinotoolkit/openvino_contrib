@echo off
setlocal enabledelayedexpansion

set "args="
set "link_args="
set "is_compile=0"

:: 1. Check if we are only compiling (-c flag)
set "check_args=%*"
for %%n in (%check_args%) do (
    if "%%n" == "-c" (
        set "is_compile=1"
    )
)

:loop
if "%~1" == "" goto :run
set "arg=%~1"

:: 2. Filter out GCC/Clang specific flags that break MSVC
if "!arg!" == "-Werror" goto :next
if "!arg!" == "-Wall" goto :next
if "!arg!" == "-m64" goto :next
if "!arg!" == "-mthreads" goto :next
if "!arg!" == "-fno-stack-protector" goto :next
if "!arg!" == "-fmessage-length=0" goto :next
if "!arg!" == "-Wno-write-strings" goto :next
if "!arg!" == "-fdiagnostics-show-option" goto :next

:: 3. Translate common CGO compiler flags
if "!arg!" == "-g" (
    set "args=!args! /Zi"
    goto :next
)
if "!arg!" == "-O2" (
    set "args=!args! /O2"
    goto :next
)
if "!arg!" == "-std=c11" (
    set "args=!args! /std:c11"
    goto :next
)
if "!arg!" == "-std=c++17" (
    set "args=!args! /std:c++17"
    goto :next
)

:: 4. Translate include and library search paths
if "!arg:~0,2!" == "-I" (
    set "args=!args! /I!arg:~2!"
    goto :next
)
if "!arg:~0,2!" == "-L" (
    set "link_args=!link_args! /LIBPATH:!arg:~2!"
    goto :next
)

:: 5. Translate library linking (e.g., -lopenvino -> openvino.lib)
if "!arg:~0,2!" == "-l" (
    set "libname=!arg:~2!"
    set "link_args=!link_args! !libname!.lib"
    goto :next
)

:: 6. Handle output flag translations
if "!arg!" == "-o" (
    if "!is_compile!" == "1" (
        set "args=!args! /Fo%~2"
    ) else (
        set "args=!args! /Fe%~2"
    )
    shift
    goto :next
)

:: Otherwise, keep the argument as-is
set "args=!args! !arg!"

:next
shift
goto :loop

:run
if not "!link_args!" == "" (
    :: echo [MSVC Wrapper] cl.exe /nologo !args! /link !link_args!
    cl.exe /nologo !args! /link !link_args!
) else (
    :: echo [MSVC Wrapper] cl.exe /nologo !args!
    cl.exe /nologo !args!
)
exit /b %errorlevel%
