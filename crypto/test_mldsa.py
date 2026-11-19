from cryptography.hazmat.primitives.asymmetric.mldsa import MLDSA44PrivateKey
from cryptography.exceptions import InvalidSignature

# 1. Generate a key pair for Agent A
private_key = MLDSA44PrivateKey.generate()
public_key = private_key.public_key()

# 2. Agent A signs a message
message = b"Agent A to Agent B: summary of task results"
signature = private_key.sign(message)
print("Message size:  ", len(message), "bytes")
print("Signature size:", len(signature), "bytes")

# 3. Agent B verifies the original message
try:
    public_key.verify(signature, message)
    print("Original message valid: True")
except InvalidSignature:
    print("Original message valid: False")

# 4. Someone changes the message: verification must fail
tampered = b"Agent A to Agent B: summary of task results!"
try:
    public_key.verify(signature, tampered)
    print("Tampered message valid: True")
except InvalidSignature:
    print("Tampered message valid: False")