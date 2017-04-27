# -*- mode: python -*-

block_cipher = None


excludes = ['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter', 'PyQt4', 'sphinx', 'cryptography']

a = Analysis(['main.py'],
             pathex=['D:\\Cloud\\Dropbox\\projects\\EDP'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=excludes,
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='main',
          debug=False,
          strip=False,
          upx=False,
          icon='resources/icons/math-multi-size.ico',
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='main')
