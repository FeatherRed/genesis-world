from pathlib import Path

p = Path(".scratch/verify.py")
s = p.read_text(encoding="utf-8-sig")

old = (
    "    n = 6000\n"
    "    q = rng.normal(size=(n, 4))\n"
    "    q /= np.linalg.norm(q, axis=1, keepdims=True)\n"
    "    q *= rng.uniform(0.5, 1.5, size=(n, 1))\n"
    "    quats.append(q)\n"
    "    n = 2000\n"
    "    roll = np.sign(rng.uniform(-1, 1, n)) * (10.0 + rng.uniform(0.05, 1.0, n))\n"
    "    pitch = rng.uniform(-70.0, 70.0, n)\n"
)
new = (
    "    n = 4000\n"
    "    q = rng.normal(size=(n, 4))\n"
    "    q /= np.linalg.norm(q, axis=1, keepdims=True)\n"
    "    q *= rng.uniform(0.5, 1.5, size=(n, 1))\n"
    "    quats.append(q)\n"
    "    # roll straddles the 10 deg boundary while pitch stays below it, so a\n"
    "    # too-strict roll predicate shows up as a false positive.\n"
    "    n = 3000\n"
    "    roll = np.sign(rng.uniform(-1, 1, n)) * (10.0 + rng.uniform(-1.0, 1.0, n))\n"
    "    pitch = rng.uniform(-8.0, 8.0, n)\n"
)
assert old in s, "roll block anchor missing"
s = s.replace(old, new, 1)

old2 = (
    "    n = 1500\n"
    "    pitch = np.sign(rng.uniform(-1, 1, n)) * (10.0 + rng.uniform(0.05, 1.0, n))\n"
    "    roll = rng.uniform(-70.0, 70.0, n)\n"
)
new2 = (
    "    n = 2500\n"
    "    pitch = np.sign(rng.uniform(-1, 1, n)) * (10.0 + rng.uniform(-1.0, 1.0, n))\n"
    "    roll = rng.uniform(-8.0, 8.0, n)\n"
)
assert old2 in s, "pitch block anchor missing"
s = s.replace(old2, new2, 1)

p.write_text(s, encoding="utf-8")
print("patched check 3")
