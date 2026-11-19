import time
import statistics
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.mldsa import MLDSA44PrivateKey
from cryptography.hazmat.primitives import serialization

RUNS = 200  # how many times to repeat each test

# Two sample agent messages: full context vs short summary
FULL_CONTEXT = ("Agent A to Agent B. Task: review the quarterly sales report. "
                "Full context: " + "The report covers revenue, costs, regional results and forecasts. " * 30).encode()
SUMMARY = b"Agent A to Agent B: Q3 revenue up 8%, costs flat, forecast raised."

SCHEMES = {"Ed25519": Ed25519PrivateKey, "ML-DSA-44": MLDSA44PrivateKey}


def time_ms(func):
    """Run func RUNS times and return the median time in milliseconds."""
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


rows = []
for name, KeyClass in SCHEMES.items():
    keygen_ms = time_ms(KeyClass.generate)
    key = KeyClass.generate()
    pub = key.public_key()
    pub_size = len(pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw))

    for label, msg in [("Full context", FULL_CONTEXT), ("Summary", SUMMARY)]:
        sig = key.sign(msg)
        sign_ms = time_ms(lambda: key.sign(msg))
        verify_ms = time_ms(lambda: pub.verify(sig, msg))
        overhead = len(sig) / len(msg) * 100
        rows.append((name, label, len(msg), len(sig), pub_size,
                     keygen_ms, sign_ms, verify_ms, overhead))

header = ("Scheme", "Message", "Msg B", "Sig B", "PubKey B",
          "Keygen ms", "Sign ms", "Verify ms", "Overhead %")
print(("{:<10}{:<14}{:>7}{:>7}{:>10}{:>11}{:>9}{:>11}{:>12}").format(*header))
for r in rows:
    print(("{:<10}{:<14}{:>7}{:>7}{:>10}{:>11.3f}{:>9.3f}{:>11.3f}{:>11.1f}%").format(*r))

# Save results so the team can use them in slides or the report
with open("crypto/signature_benchmark.csv", "w") as f:
    f.write(",".join(header) + "\n")
    for r in rows:
        f.write(",".join(str(round(x, 4)) if isinstance(x, float) else str(x) for x in r) + "\n")
print("\nSaved results to crypto/signature_benchmark.csv")