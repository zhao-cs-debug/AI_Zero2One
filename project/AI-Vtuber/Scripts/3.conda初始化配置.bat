@echo off

rem ��ȡ��ǰϵͳ�û����û���
for /f %%i in ('whoami') do set "USERNAME=%%i"

rem ����޷���ȡ�û�������Ĭ��ʹ�� "Administrator"
if "%USERNAME%"=="" set "USERNAME=Administrator"

rem ���� Conda �����ļ�·��
set "CONDARC_PATH=C:\Users\%USERNAME%\.condarc"

rem ��������ļ��Ƿ��Ѵ���
if exist "!CONDARC_PATH!" (
    echo Conda �����ļ��Ѵ��ڣ������ʼ����
    exit /b
)

rem ���� Conda �����ļ�
echo ���� Conda �����ļ�...
echo channels: > "!CONDARC_PATH!"
echo   - defaults >> "!CONDARC_PATH!"

rem ��ʼ�� Conda
echo ��ʼ�� Conda...
conda init

echo Conda �����ļ���ʼ����ɡ�
