"""Real Windows mutual exclusion between .NET setup and Python editing jobs."""
from __future__ import annotations

import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import unittest

from fpvsesh.control import acquire_run_lock


ROOT = Path(__file__).resolve().parents[1]
SHELLS = [path for name in ("powershell.exe", "pwsh.exe") if (path := shutil.which(name))]


@unittest.skipUnless(os.name == "nt" and SHELLS, "Windows PowerShell setup lock verification")
class SetupLockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fpvsesh-setup-lock-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name).resolve()
        self.app = self.folder / "application [flight] café"
        self.app.mkdir()
        self.helper = ROOT / "setup-common.ps1"
        self.flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    def command(self, shell, script, *arguments):
        return [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-File", str(script), "-AppRoot", str(self.app), "-Helper", str(self.helper), *arguments]

    def test_active_python_job_blocks_setup_before_any_package_mutation(self):
        script = self.folder / "attempt.ps1"
        script.write_text("""param([string]$AppRoot, [string]$Helper)
$ErrorActionPreference = 'Stop'
. $Helper
$guard = $null
try {
    $guard = Enter-FpvSetupLock -AppRoot $AppRoot
    [IO.File]::WriteAllText((Join-Path $AppRoot 'mutation-marker.txt'), 'setup acquired the lock')
} catch {
    [Console]::WriteLine($_.Exception.Message)
    exit 23
} finally {
    if ($guard) { $guard.Dispose() }
}
""", encoding="utf-8")
        marker = self.app / "mutation-marker.txt"
        for shell in SHELLS:
            with self.subTest(shell=Path(shell).name):
                marker.unlink(missing_ok=True)
                lock = acquire_run_lock(self.app / "cache")
                try:
                    blocked = subprocess.run(self.command(shell, script), capture_output=True, text=True,
                                             encoding="utf-8", errors="replace", timeout=30, creationflags=self.flags)
                    self.assertEqual(blocked.returncode, 23, blocked.stderr)
                    self.assertIn("Finish or cancel any active FPV Sesh", blocked.stdout)
                    self.assertFalse(marker.exists(), "Setup must stop before mutations when a job is active")
                finally:
                    lock.close()
                resumed = subprocess.run(self.command(shell, script), capture_output=True, text=True,
                                         encoding="utf-8", errors="replace", timeout=30, creationflags=self.flags)
                self.assertEqual(resumed.returncode, 0, resumed.stderr)
                self.assertTrue(marker.exists())

    def test_setup_and_its_nested_lease_block_python_until_disposed(self):
        script = self.folder / "hold.ps1"
        script.write_text("""param([string]$AppRoot, [string]$Helper, [switch]$Nested)
$ErrorActionPreference = 'Stop'
. $Helper
$guard = Enter-FpvSetupLock -AppRoot $AppRoot
$child = $null
try {
    if ($Nested) {
        $child = Enter-FpvSetupLock -AppRoot $AppRoot -ParentGuard $guard
        $guard.Dispose()
    }
    [Console]::WriteLine('LOCKED')
    [Console]::Out.Flush()
    [Console]::In.ReadLine() | Out-Null
} finally {
    if ($child) { $child.Dispose() }
    $guard.Dispose()
}
""", encoding="utf-8")
        for shell in SHELLS:
            for nested in (False, True):
                with self.subTest(shell=Path(shell).name, nested=nested):
                    process = subprocess.Popen(self.command(shell, script, *(["-Nested"] if nested else [])),
                                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                               text=True, encoding="utf-8", errors="replace", creationflags=self.flags)
                    messages = queue.Queue()
                    reader = threading.Thread(target=lambda: messages.put(process.stdout.readline()), daemon=True)
                    reader.start()
                    try:
                        self.assertEqual(messages.get(timeout=30).strip(), "LOCKED")
                        with self.assertRaisesRegex(RuntimeError, "Another FPV Sesh job"):
                            acquire_run_lock(self.app / "cache")
                        _, error = process.communicate("\n", timeout=15)
                        self.assertEqual(process.returncode, 0, error)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait(timeout=10)
                        for pipe in (process.stdin, process.stdout, process.stderr):
                            if pipe is not None:
                                pipe.close()
                        reader.join(timeout=5)
                    unlocked = acquire_run_lock(self.app / "cache")
                    unlocked.close()


if __name__ == "__main__":
    unittest.main()
