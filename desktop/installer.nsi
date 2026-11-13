; VEviaH setup (2026-11-13, NSIS; built by build_release.sh: makensis -DVERSION= -DSRC=<staged folder> -DOUTFILE=).
; Per-user install, no admin prompt: %LOCALAPPDATA%\Programs\VEviaH (fixed). Options: desktop shortcut (on), start
; with Windows (off), run when done. Updating over an installed copy closes VEviaH first (its MCP servers too: they run
; on VEviaH.exe and lock it) and keeps the settings, chat history, API keys and the voice engine (all outside the
; install folder). Uninstall: our entry out of every connected AI app (VEviaH.exe --uninstall-cleanup), program and
; shortcuts gone; settings / history / voice engine only if the user says yes (default no).
Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"

Name "VEviaH"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\VEviaH"
RequestExecutionLevel user
ShowInstDetails nevershow      ; progress bar + current file only (the details button could not be folded back)
ShowUninstDetails nevershow
SetCompressor /SOLID lzma
BrandingText "VEviaH ${VERSION}"
VIProductVersion "${VERSION}.0"
VIAddVersionKey /LANG=0 "ProductName" "VEviaH"
VIAddVersionKey /LANG=0 "FileVersion" "${VERSION}"
VIAddVersionKey /LANG=0 "ProductVersion" "${VERSION}"
VIAddVersionKey /LANG=0 "FileDescription" "VEviaH setup"
VIAddVersionKey /LANG=0 "LegalCopyright" "VEviaH"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\VEviaH"

!define MUI_ICON "${SRC}\resources\app\icon.ico"
!define MUI_UNICON "${SRC}\resources\app\icon.ico"
!define MUI_ABORTWARNING
!define MUI_COMPONENTSPAGE_NODESC
!define MUI_FINISHPAGE_RUN "$INSTDIR\VEviaH.exe"
!define MUI_LANGDLL_ALLLANGUAGES

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "TradChinese"
!insertmacro MUI_LANGUAGE "English"
!insertmacro MUI_LANGUAGE "Japanese"

LangString SecApp ${LANG_TRADCHINESE} "VEviaH（必要）"
LangString SecApp ${LANG_ENGLISH} "VEviaH (required)"
LangString SecApp ${LANG_JAPANESE} "VEviaH（必須）"
LangString SecDesk ${LANG_TRADCHINESE} "建立桌面捷徑"
LangString SecDesk ${LANG_ENGLISH} "Create a desktop shortcut"
LangString SecDesk ${LANG_JAPANESE} "デスクトップにショートカットを作成"
LangString SecAuto ${LANG_TRADCHINESE} "Windows 開機時啟動 VEviaH"
LangString SecAuto ${LANG_ENGLISH} "Launch VEviaH when Windows starts"
LangString SecAuto ${LANG_JAPANESE} "Windows の起動時に VEviaH を起動"
LangString Closing ${LANG_TRADCHINESE} "正在關閉 VEviaH…"
LangString Closing ${LANG_ENGLISH} "Closing VEviaH…"
LangString Closing ${LANG_JAPANESE} "VEviaH を終了しています…"
LangString AskData ${LANG_TRADCHINESE} "要一併刪除 VEviaH 的設定、聊天紀錄、API key、語音引擎（約 9 GB）和你加入的聲音嗎？$\n$\n選「否」會保留它們，之後重新安裝可以直接沿用。"
LangString AskData ${LANG_ENGLISH} "Also delete VEviaH's settings, chat history, API keys, the voice engine (about 9 GB) and the voices you added?$\n$\nChoose No to keep them for a later reinstall."
LangString AskData ${LANG_JAPANESE} "VEviaH の設定、チャット履歴、API キー、音声エンジン（約 9 GB）、追加した音声も削除しますか？$\n$\n「いいえ」を選ぶと保持され、再インストール時にそのまま使えます。"

Function .onInit
  !insertmacro MUI_LANGDLL_DISPLAY
FunctionEnd
Function un.onInit
  !insertmacro MUI_UNGETLANGUAGE
FunctionEnd

; VEviaH.exe runs as the app and as the MCP servers AI apps start; all of them hold the exe and its files. Only the
; copies running from this install folder are closed (another copy elsewhere, e.g. a zip, keeps running).
; Through WMI (Win32_Process.ExecutablePath): this setup is a 32-bit program, so it starts the 32-bit PowerShell, whose
; Get-Process cannot read a 64-bit process's Path (2026-11-13: nothing was closed and VEviaH.exe could not be written).
; A path may come in its 8.3 short form (C:\Users\NAME~1\... when started through %TEMP%), so both forms count.
!macro CloseVEviaH
  DetailPrint "$(Closing)"
  GetFullPathName /SHORT $R9 "$INSTDIR"
  StrCmp $R9 "" 0 +2
    StrCpy $R9 "$INSTDIR"
  nsExec::Exec `powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $$_.Name -eq 'VEviaH.exe' -and $$_.ExecutablePath -and ($$_.ExecutablePath.StartsWith('$INSTDIR\', 'OrdinalIgnoreCase') -or $$_.ExecutablePath.StartsWith('$R9\', 'OrdinalIgnoreCase')) } | ForEach-Object { Stop-Process -Id $$_.ProcessId -Force -ErrorAction SilentlyContinue }"`
  Pop $0
  Sleep 1500
!macroend

Section "!$(SecApp)" SecMain
  SectionIn RO
  !insertmacro CloseVEviaH
  ; an update starts from an empty program folder, so files a newer version dropped do not linger (2026-11-13: an old
  ; model and WSL link files stayed); only in a folder that holds VEviaH.exe. Settings, chats and voices live elsewhere.
  ${If} ${FileExists} "$INSTDIR\VEviaH.exe"
    RMDir /r "$INSTDIR\resources"
    RMDir /r "$INSTDIR\backend"
    RMDir /r "$INSTDIR\locales"
    Delete "$INSTDIR\*.*"
  ${EndIf}
  SetOutPath "$INSTDIR"
  File /r "${SRC}\*.*"
  WriteUninstaller "$INSTDIR\Uninstall VEviaH.exe"
  CreateShortcut "$SMPROGRAMS\VEviaH.lnk" "$INSTDIR\VEviaH.exe" "" "$INSTDIR\resources\app\icon.ico"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "VEviaH"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINST_KEY}" "Publisher" "VEviaH"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\resources\app\icon.ico"
  WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" '"$INSTDIR\Uninstall VEviaH.exe"'
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1
SectionEnd

Section "$(SecDesk)" SecDesktop
  CreateShortcut "$DESKTOP\VEviaH.lnk" "$INSTDIR\VEviaH.exe" "" "$INSTDIR\resources\app\icon.ico"
SectionEnd

Section /o "$(SecAuto)" SecAutostart
  ExecWait '"$INSTDIR\VEviaH.exe" --autostart-on'
SectionEnd

Section "Uninstall"
  ; only a folder this setup made (the uninstaller and the exe are both there)
  ${IfNot} ${FileExists} "$INSTDIR\VEviaH.exe"
    Abort
  ${EndIf}
  !insertmacro CloseVEviaH
  ExecWait '"$INSTDIR\VEviaH.exe" --uninstall-cleanup'
  Delete "$DESKTOP\VEviaH.lnk"
  Delete "$SMPROGRAMS\VEviaH.lnk"
  DeleteRegKey HKCU "${UNINST_KEY}"
  RMDir /r "$INSTDIR\resources"
  RMDir /r "$INSTDIR\backend"
  RMDir /r "$INSTDIR\locales"
  Delete "$INSTDIR\*.*"
  RMDir "$INSTDIR"
  MessageBox MB_YESNO|MB_ICONQUESTION|MB_DEFBUTTON2 "$(AskData)" /SD IDNO IDNO keep
    RMDir /r "$APPDATA\veviah"
    RMDir /r "$LOCALAPPDATA\veviah"
  keep:
SectionEnd
