"""Optional lifetime management for a model server this process starts itself.

Never accepts an existing PID or controls a shared/remote model server. Importing
this module starts nothing. GPU memory reclamation requires separate telemetry.
"""
import os
import signal
import subprocess
import time
import urllib.parse
import urllib.request

from .contract import ContractError


class OwnedSetupServer:
    def __init__(self, command):
        if not isinstance(command, list) or not command or any(type(s) is not str or not s for s in command):
            raise ContractError('TRUSTED_SERVER_COMMAND_REQUIRED')
        self.command = command[:]
        self.process = None
        self._closed = False

    def __enter__(self):
        if self.process is not None or self._closed:
            raise ContractError('SERVER_ALREADY_STARTED')
        options = {'stdin': subprocess.DEVNULL, 'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL}
        if os.name == 'nt':
            options['creationflags'] = subprocess.CREATE_NO_WINDOW
        else:
            options['start_new_session'] = True
        self.process = subprocess.Popen(self.command, **options)
        return self

    def wait_ready(self, base_url, timeout_s=60):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'} or not 0 < timeout_s <= 60:
            raise ContractError('OWNED_LOCAL_ENDPOINT_REQUIRED')
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                raise ContractError('OWNED_SERVER_EXITED')
            try:
                with urllib.request.urlopen(base_url.rstrip('/') + '/models', timeout=1) as response:
                    if response.status == 200:
                        return
            except (OSError, ValueError):
                time.sleep(0.1)
        raise ContractError('OWNED_SERVER_STARTUP_TIMEOUT')

    def close(self):
        process = self.process
        if process is None or self._closed:
            return
        self._closed = True
        if process.poll() is not None:
            # An exited PID may be recycled; never signal a stale process ID.
            return
        if os.name != 'nt':
            # start_new_session makes the new child's PID its own group ID.
            # The group may still contain workers after the server parent exits.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            # Windows tests cover a single owned child. Full vLLM worker-tree
            # cleanup is Linux-only; do not claim Windows GPU worker cleanup.
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

    def __exit__(self, *args):
        self.close()
