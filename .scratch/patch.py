from pathlib import Path

p = Path(".scratch/verify.py")
s = p.read_text(encoding="utf-8-sig")

old1 = "from genesis.utils.geom import quat_to_xyz\n"
new1 = (
    "import genesis as gs\n"
    "from genesis.utils.geom import quat_to_xyz\n"
    "\n"
    "# gs.init() sets EPS from np.finfo(np.float32).eps on a float32 backend and the\n"
    "# numba oracle in geom.py reads it, so mirror it here without a simulator.\n"
    "gs.EPS = float(np.finfo(np.float32).eps)\n"
)
assert old1 in s, "import anchor missing"
s = s.replace(old1, new1, 1)

old2 = (
    '    zero = f32(np.zeros((1, 12)))\n'
    '\n'
    '    def reward_of(st, actions=zero, last=zero):\n'
    '        return float(advance_reward(exe, st, actions, last, 1)["reward"][0])\n'
)
new2 = (
    '    zero = f32(np.zeros((1, 12)))\n'
    '    adv1 = K.build_advance(1, max_episode_length=1000)\n'
    '\n'
    '    def reward_of(st, actions=zero, last=zero):\n'
    '        return float(advance_reward(adv1, st, actions, last, 1)["reward"][0])\n'
)
assert old2 in s, "reward_of anchor missing"
s = s.replace(old2, new2, 1)

p.write_text(s, encoding="utf-8")
print("patched")
