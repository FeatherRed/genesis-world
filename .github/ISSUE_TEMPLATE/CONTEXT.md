# Intent

GitHub issue forms used when filing issues against `genesis-world`, plus the issue-chooser config.

# Files

- `config.yml` — disables blank issues; routes support/questions to GitHub Discussions.
- `1-bug-report.yml` — "Bug report" form (`[Bug]: `, labels `bug`, `triage-needed`): description, reproduction script, expected behavior, screenshots, logs, environment, version/commit.
- `2-feature-request.yml` — "Feature Request" form (`[Feature]: `, labels `feature`, `enhancement`, `triage-needed`): proposal, motivation (user story), benefit, acceptance criteria.
- `3-documentation.yml` — "Documentation Issue" form (`[Docs]: `, labels `documentation`, `triage-needed`): affected page, what is wrong/missing, suggested improvement.
  Documentation issues are tracked here even though the pages live in the separate `genesis-doc` repository.

# See Also

- `../pull_request_template.md` → PR-side counterpart to these issue forms.
- `../contributing/PULL_REQUESTS.md` → the "reporting bugs / suggesting features" guidance these forms realize.
