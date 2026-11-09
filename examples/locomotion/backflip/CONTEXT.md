# examples/locomotion/backflip

## Intent
Holds the pre-trained policy checkpoints for the Go2 single and double backflips, plus a readme describing how to run them.

## API Surface
- `single.pt`, `double.pt` → torch `state_dict` checkpoints (~791 KB each) for a fixed 60→512→256→128→12 ELU MLP policy.
- `readme.md` → usage commands and a pointer to the external advanced training repo (`https://github.com/ziyanx02/Genesis-backflip`).

## Constraints
- These files are data, not code: never edit, regenerate, or "fix" them; git history tracks provenance.
- The network architecture is fixed in `go2_backflip.py` and must match the checkpoint shapes.

## See Also
- `../go2_backflip.py` → loads these checkpoints with `torch.load(..., weights_only=True)` and plays them back.
