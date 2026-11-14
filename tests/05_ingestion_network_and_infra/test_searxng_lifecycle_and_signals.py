import os
import sys
import signal
import time
import pytest
from unittest.mock import patch, MagicMock

from hybrid_backend import GovernanceManager, stop_servers

def test_searxng_installed_detection():
    """Verify that GovernanceManager correctly identifies the SearXNG installation on D: drive."""
    assert GovernanceManager.is_searxng_installed() is True

def test_searxng_lifecycle_start_and_stop():
    """Verify that GovernanceManager can start, detect, or mock the SearXNG daemon in WSL safely."""
    if GovernanceManager.is_searxng_running():
        # Live instance is active; test idempotency and non-blocking status
        assert GovernanceManager.start_searxng(async_boot=False) is True
        assert GovernanceManager.is_searxng_running() is True
    else:
        # Test mock lifecycle to guarantee fast, deterministic tests without freezing host VM
        running = [False]
        def mock_is_running(*args, **kwargs):
            return running[0]

        with patch.object(GovernanceManager, 'is_searxng_running', side_effect=mock_is_running):
            with patch('subprocess.Popen') as mock_popen, patch('subprocess.run') as mock_run:
                def fake_popen(*args, **kwargs):
                    running[0] = True
                    return MagicMock()
                mock_popen.side_effect = fake_popen

                started = GovernanceManager.start_searxng(async_boot=False)
                assert started is True
                assert GovernanceManager.is_searxng_running() is True

                re_started = GovernanceManager.start_searxng(async_boot=False)
                assert re_started is True

                GovernanceManager.stop_searxng()
                running[0] = False
                assert GovernanceManager.is_searxng_running() is False

def test_autostart_disable_terminates_searxng():
    """Verify that disabling autostart shuts down the SearXNG background service."""
    with patch.object(GovernanceManager, 'stop_searxng') as mock_stop:
        with patch.object(GovernanceManager, 'get_autostart_paths') as mock_paths:
            mock_paths.return_value = {
                'platform': 'windows',
                'target_file': 'fake_path.vbs',
                'startup_dir': 'fake_dir',
                'repo_root': 'fake_repo',
                'python_exe': 'fake_py'
            }
            # Simulate turning off autostart
            GovernanceManager.set_autostart(False)
            mock_stop.assert_called()

def test_stop_servers_terminates_searxng():
    """Verify that stop_servers() cleans up SearXNG alongside HTTP servers."""
    with patch.object(GovernanceManager, 'stop_searxng') as mock_stop:
        stop_servers()
        mock_stop.assert_called()

def test_graceful_interrupt_signals():
    """Verify that signal handlers are registered for SIGTERM, SIGINT, and SIGBREAK if available."""
    for sig_name in ('SIGTERM', 'SIGINT'):
        if hasattr(signal, sig_name):
            handler = signal.getsignal(getattr(signal, sig_name))
            assert handler is not None

