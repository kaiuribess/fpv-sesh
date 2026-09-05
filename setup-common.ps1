# Shared byte-zero lock for dependency/tool changes and Python editing jobs.
# Dot-source this helper, acquire before mutation, and Dispose() in finally.
if (-not ('FpvSesh.SetupGuard' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.IO;

namespace FpvSesh {
    public sealed class SetupGuard : IDisposable {
        private sealed class SharedLock {
            internal readonly string Root;
            internal readonly FileStream Stream;
            internal int References;
            internal SharedLock(string root, FileStream stream) {
                Root = root;
                Stream = stream;
                References = 1;
            }
        }

        private readonly SharedLock shared;
        private bool disposed;

        private SetupGuard(SharedLock sharedLock) { shared = sharedLock; }

        public static SetupGuard Acquire(string appRoot, SetupGuard parent) {
            string root = Path.GetFullPath(appRoot);
            if (root.Length > Path.GetPathRoot(root).Length)
                root = root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            if (parent != null) {
                // A synchronous nested installer explicitly borrows a live
                // lease. It cannot unlock the parent's operation prematurely.
                lock (parent.shared) {
                    if (parent.disposed || parent.shared.References < 1 ||
                        !String.Equals(root, parent.shared.Root, StringComparison.OrdinalIgnoreCase)) {
                        throw new InvalidOperationException("The parent setup guard is closed or belongs to another application.");
                    }
                    parent.shared.References++;
                    return new SetupGuard(parent.shared);
                }
            }
            string cache = Path.Combine(root, "cache");
            Directory.CreateDirectory(cache);
            FileStream stream = new FileStream(Path.Combine(cache, "run.lock"), FileMode.OpenOrCreate,
                                               FileAccess.ReadWrite, FileShare.ReadWrite);
            try {
                // Do not rewrite the byte while an editing process owns it.
                if (stream.Length == 0) {
                    stream.WriteByte(48);
                    stream.Flush();
                }
                stream.Lock(0, 1);
                return new SetupGuard(new SharedLock(root, stream));
            } catch {
                stream.Dispose();
                throw;
            }
        }

        public void Dispose() {
            lock (shared) {
                if (disposed) return;
                disposed = true;
                shared.References--;
                if (shared.References == 0) shared.Stream.Dispose();
            }
        }
    }
}
'@
}

function Enter-FpvSetupLock {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [object]$ParentGuard = $null
    )
    try {
        return [FpvSesh.SetupGuard]::Acquire($AppRoot, $ParentGuard)
    } catch {
        throw 'Setup could not acquire exclusive access. Finish or cancel any active FPV Sesh edit or flight analysis, close another running setup, and try again. The application folder must also be writable.'
    }
}
