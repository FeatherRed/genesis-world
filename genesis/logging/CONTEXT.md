# genesis/logging/ — Logger

## Intent
Genesis's console logging layer: a themed formatter and logger that render the `~kind<...>~` markup of `genesis/styles.py`, plus a helper that prints an in-place elapsed-time readout.

## API Surface
- `__init__.py` — re-exports `Logger`.
- `logger.py` — `GenesisFormatter` (maps levels to colors, renders markup + theme prefix) and `Logger` (wraps stdlib `logging`, owns `last_output`, exposes `banner`, `shutdown`, and the colored theme set at construction).
- `time_elapser.py` — `TimeElapser`: a `with`-statement context manager that periodically re-renders the current log line with elapsed time (rate-limited, threaded).

## Constraints
The logger theme (`THEME.dark/light/raw`) is set once in `Logger.__init__`; `raw` is used for CI/agents/log files.
`gs.logger` is the single global logger instance created by `gs.init(...)`.

## Design Decisions
Messages embed markup (`~<value>~`, `~~<name>~~`, `~~~<uid>~~~`, `~~~~<title>~~~~`) that `styles.STYLE.markup` renders per theme.

## See Also
- `../styles.py` — theme, colors, formats, markup parser.
- `../__init__.py` — creates and owns the global logger.

## Status
Complete.
