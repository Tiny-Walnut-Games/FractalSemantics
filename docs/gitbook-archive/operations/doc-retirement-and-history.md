# Operations: Documentation Lifecycle

Use this policy to keep documentation clean and current.

## Keep / Update / Remove

- Keep active docs tied to current workflows.
- Update long-lived guides when flags or behavior changes.
- Remove one-off fix summaries when superseded.

## Recover from Git History

```bash
git log --diff-filter=D --name-status -- "*.md"
git log --follow -- path/to/file.md
git show <commit_sha>:path/to/file.md
git log -S "search phrase" -- "*.md"
```

## Retirement Checklist

1. Confirm current docs already cover replacement behavior.
2. Remove obsolete docs from tree.
3. Verify no broken references remain.
4. Add recovery pointers in relevant docs if needed.
