"""
Encrypted embedding storage using AES-256-GCM.

Encrypts face embeddings at rest with authenticated encryption.
Supports key rotation and pluggable key providers.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional, Protocol

import msgpack
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from facepipe.config.settings import get_settings, KeyProviderType
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


class KeyProvider(Protocol):
    """Protocol for encryption key providers."""

    def get_key(self) -> bytes:
        """Return a 32-byte encryption key."""
        ...


class EnvKeyProvider:
    """Load encryption key from environment variable."""

    def __init__(self, env_var: str = "FR_ENCRYPTION_KEY") -> None:
        self._env_var = env_var

    def get_key(self) -> bytes:
        key_str = os.environ.get(self._env_var, "")
        if not key_str:
            # Generate a default key for development (NOT for production)
            logger.warning("no_encryption_key_set", action="using_default_dev_key")
            key_str = base64.b64encode(os.urandom(32)).decode()
            os.environ[self._env_var] = key_str
        return base64.b64decode(key_str)


class FileKeyProvider:
    """Load encryption key from a file."""

    def __init__(self, path: str) -> None:
        self._path = path

    def get_key(self) -> bytes:
        with open(self._path, "rb") as f:
            return f.read(32)


class EncryptedEmbeddingStore:
    """AES-256-GCM encrypted storage for face embeddings.

    Each embedding is encrypted individually with a unique nonce.
    Storage format: MessagePack-serialized dict with nonce + ciphertext.

    Args:
        storage_dir: Directory to store encrypted embedding files.
        key_provider: Key provider implementation. If None, auto-detected from facepipe.config.
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        key_provider: Optional[KeyProvider] = None,
    ) -> None:
        settings = get_settings()

        if storage_dir is None:
            storage_dir = str(settings.data_dir / "embeddings")
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        if key_provider is None:
            if settings.storage.key_provider == KeyProviderType.FILE:
                key_provider = FileKeyProvider(settings.storage.key_file_path)
            else:
                key_provider = EnvKeyProvider()

        self._key_provider = key_provider
        self._aesgcm: Optional[AESGCM] = None

    def _get_cipher(self) -> AESGCM:
        """Lazily initialize the AES-GCM cipher."""
        if self._aesgcm is None:
            key = self._key_provider.get_key()
            if len(key) != 32:
                raise ValueError(f"Encryption key must be 32 bytes, got {len(key)}")
            self._aesgcm = AESGCM(key)
        return self._aesgcm

    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        """Encrypt a single embedding.

        Args:
            embedding: numpy array to encrypt.

        Returns:
            MessagePack-serialized bytes containing nonce + ciphertext.
        """
        cipher = self._get_cipher()
        nonce = os.urandom(12)  # 96-bit nonce for AES-GCM
        plaintext = embedding.astype(np.float32).tobytes()
        ciphertext = cipher.encrypt(nonce, plaintext, None)

        return msgpack.packb({
            "nonce": nonce,
            "ciphertext": ciphertext,
            "dim": embedding.shape[0] if embedding.ndim == 1 else embedding.shape,
            "dtype": "float32",
        })

    def decrypt_embedding(self, encrypted: bytes) -> np.ndarray:
        """Decrypt a single embedding.

        Args:
            encrypted: MessagePack-serialized encrypted data.

        Returns:
            Decrypted numpy array.
        """
        cipher = self._get_cipher()
        data = msgpack.unpackb(encrypted, raw=True)

        nonce = data[b"nonce"]
        ciphertext = data[b"ciphertext"]
        dim = data[b"dim"]

        plaintext = cipher.decrypt(nonce, ciphertext, None)
        embedding = np.frombuffer(plaintext, dtype=np.float32)

        if isinstance(dim, int):
            embedding = embedding.reshape(dim)
        else:
            embedding = embedding.reshape(tuple(dim))

        return embedding

    def save_identity_embeddings(
        self,
        identity_id: str,
        embeddings: list[np.ndarray],
        model_version: str = "",
    ) -> None:
        """Save all embeddings for an identity to disk (encrypted).

        Args:
            identity_id: Identity UUID.
            embeddings: List of embedding arrays.
            model_version: Model version tag for embedding versioning.
        """
        identity_dir = self._storage_dir / identity_id
        identity_dir.mkdir(parents=True, exist_ok=True)

        # Encrypt each embedding
        encrypted_data = []
        for emb in embeddings:
            encrypted_data.append(self.encrypt_embedding(emb))

        # Save as a single MessagePack file
        payload = msgpack.packb({
            "identity_id": identity_id,
            "model_version": model_version,
            "count": len(embeddings),
            "embeddings": encrypted_data,
        })

        # Atomic write: write to temp, then rename
        target = identity_dir / "embeddings.enc"
        temp = identity_dir / "embeddings.enc.tmp"
        with open(temp, "wb") as f:
            f.write(payload)
        temp.rename(target)

        logger.debug("embeddings_saved", identity_id=identity_id, count=len(embeddings))

    def load_identity_embeddings(self, identity_id: str) -> list[np.ndarray]:
        """Load all embeddings for an identity from disk (decrypted).

        Args:
            identity_id: Identity UUID.

        Returns:
            List of decrypted embedding arrays.
        """
        target = self._storage_dir / identity_id / "embeddings.enc"
        if not target.exists():
            return []

        with open(target, "rb") as f:
            payload = msgpack.unpackb(f.read(), raw=True)

        encrypted_list = payload[b"embeddings"]
        return [self.decrypt_embedding(enc) for enc in encrypted_list]

    def delete_identity(self, identity_id: str) -> bool:
        """Delete all encrypted data for an identity.

        Args:
            identity_id: Identity UUID.

        Returns:
            True if deleted, False if not found.
        """
        identity_dir = self._storage_dir / identity_id
        if not identity_dir.exists():
            return False

        import shutil
        shutil.rmtree(identity_dir)
        logger.info("identity_embeddings_deleted", identity_id=identity_id)
        return True

    def rotate_key(self, new_key_provider: KeyProvider) -> int:
        """Re-encrypt all embeddings with a new key.

        Args:
            new_key_provider: The new key provider.

        Returns:
            Number of identities re-encrypted.
        """
        count = 0
        old_cipher = self._get_cipher()

        # Load all with old key
        for identity_dir in self._storage_dir.iterdir():
            if not identity_dir.is_dir():
                continue

            identity_id = identity_dir.name
            embeddings = self.load_identity_embeddings(identity_id)

            if embeddings:
                # Switch to new key
                self._aesgcm = None
                self._key_provider = new_key_provider
                self.save_identity_embeddings(identity_id, embeddings)
                count += 1

        logger.info("key_rotation_complete", identities_rotated=count)
        return count
