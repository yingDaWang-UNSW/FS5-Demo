# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['runFS5GUIDemo.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries+[('G:/My Drive/sourceCodes/fs5ydw-main/tmp/warpcache/wp_warpOverrides.fs5WarpOverrides_27950ef', './tmp/warpcache/wp_warpOverrides.fs5WarpOverrides_27950ef', 'BINARY')],
    a.datas,
    [],
    name='runFS5GUIDemo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
