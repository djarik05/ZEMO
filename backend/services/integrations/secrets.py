import base64
import hashlib
import os
import secrets
from typing import Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


class SecretManager:
    def __init__(self):
        settings = get_settings()
        self._fernet: Optional[Fernet] = None
        self._init_fernet(settings.secret_key)

    def _init_fernet(self, secret_key: str) -> None:
        if not secret_key:
            logger.warning("secret_key not configured, secret encryption disabled")
            return

        salt = b"zemo-salt-v1"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(secret_key.encode()))
        self._fernet = Fernet(key)

    def encrypt(self, value: str) -> str:
        if not self._fernet:
            raise RuntimeError("Secret encryption not configured")
        encrypted = self._fernet.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt(self, encrypted_value: str) -> str:
        if not self._fernet:
            raise RuntimeError("Secret encryption not configured")
        decoded = base64.urlsafe_b64decode(encrypted_value.encode())
        decrypted = self._fernet.decrypt(decoded)
        return decrypted.decode()

    def is_configured(self) -> bool:
        return self._fernet is not None


class SecretStore:
    def __init__(self):
        self._secrets: dict[str, str] = {}
        self._secret_manager = SecretManager()
        self._use_encryption = self._secret_manager.is_configured()

    def generate_ref(self) -> str:
        return secrets.token_urlsafe(16)

    def store(self, ref: str, value: str) -> None:
        if self._use_encryption:
            self._secrets[ref] = self._secret_manager.encrypt(value)
        else:
            logger.warning("storing secret unencrypted - SECRET_KEY not configured")
            self._secrets[ref] = value

    def retrieve(self, ref: str) -> Optional[str]:
        encrypted = self._secrets.get(ref)
        if encrypted is None:
            return None
        if self._use_encryption:
            try:
                return self._secret_manager.decrypt(encrypted)
            except Exception as e:
                logger.error("failed_to_decrypt_secret", ref=ref, error=str(e))
                return None
        return encrypted

    def delete(self, ref: str) -> bool:
        if ref in self._secrets:
            del self._secrets[ref]
            return True
        return False

    def rotate(self, ref: str, new_value: str) -> bool:
        if ref in self._secrets:
            self.store(ref, new_value)
            return True
        return False


_secret_store: Optional[SecretStore] = None


def get_secret_store() -> SecretStore:
    global _secret_store
    if _secret_store is None:
        _secret_store = SecretStore()
    return _secret_store