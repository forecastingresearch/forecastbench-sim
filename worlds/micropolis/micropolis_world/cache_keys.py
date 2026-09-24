"""Prompt-cache digest extracted unchanged from Fabio Rocha's knowledge runner."""
import base64
import hashlib

def prompt_hash(prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")[:8]
