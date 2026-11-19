"""Step 1: sign and verify one agent message with Ed25519 (classical baseline)."""
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.exceptions import InvalidSignature

# A sample message one agent might send to another
message = b"Agent A to Agent B: the answer is Band-e Amir National Park."

# The sending agent creates a key pair
private_key = Ed25519PrivateKey.generate()
public_key = private_key.public_key()

# Sign the message
signature = private_key.sign(message)
print("Message size:  ", len(message), "bytes")
print("Signature size:", len(signature), "bytes")

# The receiving agent verifies it
public_key.verify(signature, message)
print("Original message: verified")

# An attacker changes one word
tampered = message.replace(b"Amir", b"Amor")
try:
    public_key.verify(signature, tampered)
    print("Tampered message: verified (this should not happen!)")
except InvalidSignature:
    print("Tampered message: rejected")