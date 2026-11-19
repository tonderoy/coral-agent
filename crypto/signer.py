"""Sign and verify agent messages with Ed25519 or ML-DSA-44.

Roy's agents can call wrap() before sending and unwrap() after receiving.
"""
import base64
import json
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.mldsa import MLDSA44PrivateKey

SCHEMES = {"ed25519": Ed25519PrivateKey, "mldsa44": MLDSA44PrivateKey}


def new_agent_keys(scheme):
    """Create a key pair for one agent. Returns (private_key, public_key)."""
    private_key = SCHEMES[scheme].generate()
    return private_key, private_key.public_key()


def wrap(sender, text, private_key, scheme):
    """Turn a message into a signed JSON envelope ready to send."""
    body = {"sender": sender, "scheme": scheme, "text": text}
    to_sign = json.dumps(body, sort_keys=True).encode()
    body["signature"] = base64.b64encode(private_key.sign(to_sign)).decode()
    return json.dumps(body)


def unwrap(envelope, trusted_public_keys):
    """Check the signature. Returns the text if valid, raises an error if not."""
    body = json.loads(envelope)
    signature = base64.b64decode(body.pop("signature"))
    public_key = trusted_public_keys[body["sender"]]  # only keys we already trust
    to_check = json.dumps(body, sort_keys=True).encode()
    public_key.verify(signature, to_check)  # raises InvalidSignature if changed
    return body["text"]


if __name__ == "__main__":
    for scheme in SCHEMES:
        priv, pub = new_agent_keys(scheme)
        trusted = {"agent_a": pub}
        text = "Q3 revenue up 8%, costs flat, forecast raised."
        envelope = wrap("agent_a", text, priv, scheme)
        print(f"\n{scheme}: plain text {len(text)} B, signed envelope {len(envelope)} B")
        print("  Valid message accepted:", unwrap(envelope, trusted) == text)
        tampered = envelope.replace("8%", "9%")
        try:
            unwrap(tampered, trusted)
            print("  Tampered message accepted: True (PROBLEM)")
        except InvalidSignature:
            print("  Tampered message rejected: True")