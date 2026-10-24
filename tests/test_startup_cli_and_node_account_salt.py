import os
import sys
import tempfile
import warnings
import unittest
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.safety.hardware_fingerprint import (
    resolve_node_salt_path,
    load_or_create_node_salt,
    is_node_account_initialized,
    verify_node_account,
    get_active_node_account_info,
    is_creator_initialized,
    verify_sole_creator
)
from src.data.canonical_projection import CanonicalProjector


class TestStartupCliAndNodeAccountSalt(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_flashdrive_salt = os.path.join(self.temp_dir.name, "removable_usb", "node_salt.key")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_flashdrive_salt_creation_and_loading(self):
        # Verify resolution to custom external path
        resolved = resolve_node_salt_path(salt_path=self.mock_flashdrive_salt)
        self.assertEqual(resolved, os.path.abspath(self.mock_flashdrive_salt))

        # First load should create the 32-byte salt file on the mock flashdrive
        self.assertFalse(os.path.exists(self.mock_flashdrive_salt))
        salt_bytes = load_or_create_node_salt(salt_path=self.mock_flashdrive_salt)
        self.assertEqual(len(salt_bytes), 32)
        self.assertTrue(os.path.exists(self.mock_flashdrive_salt))

        # Subsequent load must return identical salt
        reloaded_salt = load_or_create_node_salt(salt_path=self.mock_flashdrive_salt)
        self.assertEqual(salt_bytes, reloaded_salt)

        # Verification of account authentication with flashdrive salt
        self.assertTrue(is_node_account_initialized(salt_path=self.mock_flashdrive_salt))
        self.assertTrue(verify_node_account(account="alice", passphrase="secret_password", salt_path=self.mock_flashdrive_salt))

    def test_canonical_projector_binds_external_salt(self):
        # Instantiate CanonicalProjector with external flashdrive salt path
        projector = CanonicalProjector(dim=32, k=3, salt_path=self.mock_flashdrive_salt, account="traveler_node")
        self.assertTrue(hasattr(projector, "_hmac_key"))
        self.assertEqual(len(projector._hmac_key), 32)

        # Check projection tensor deterministic generation
        result = projector.project_text_to_state("Testing nomadic flashdrive salt resonance")
        tensor = result['state']
        self.assertEqual(tensor.shape, (1, 32))
        self.assertIn('entropy', result)


    def test_tombstoned_creator_systems_deprecation_warnings(self):
        # Verify deprecation warnings when calling tombstoned creator functions
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            init_status = is_creator_initialized()
            self.assertIsInstance(init_status, bool)
            self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught_warnings))

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            auth_status = verify_sole_creator("legacy_passphrase")
            self.assertTrue(auth_status)
            self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught_warnings))

    def test_account_info_metadata(self):
        info = get_active_node_account_info(salt_path=self.mock_flashdrive_salt, account="nomad_1")
        self.assertEqual(info["account"], "nomad_1")
        self.assertEqual(info["salt_path"], os.path.abspath(self.mock_flashdrive_salt))
        self.assertTrue(info["creator_system_tombstoned"])
        self.assertIn("hardware_fingerprint", info)


if __name__ == "__main__":
    unittest.main()
