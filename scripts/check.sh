#!/usr/bin/env bash
# Validate the committed data, manifest and charts. Run before committing.
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
note() { printf '%-28s %s\n' "$1" "$2"; }

# 1. every .jsonl parses
bad=$(python3 - <<'PY'
import json, pathlib, sys
bad = []
for p in sorted(pathlib.Path("data").glob("*.jsonl")):
    for i, line in enumerate(p.open(encoding="utf-8", errors="replace"), 1):
        if not line.strip().startswith("{"):
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            bad.append(f"{p}:{i}: {e}")
for p in sorted(pathlib.Path("data").glob("*.json")):
    try:
        json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        bad.append(f"{p}: {e}")
print("\n".join(bad))
PY
)
if [ -n "$bad" ]; then echo "$bad"; note "json" "FAIL"; fail=1; else note "json" "ok"; fi

if python3 scripts/gsq_summary.py >/dev/null; then note "September data" "ok"
else note "September data" "FAIL"; fail=1; fi

# 2. scripts parse
syn=0
for f in scripts/*.sh; do bash -n "$f" || { note "syntax $f" "FAIL"; syn=1; }; done
for f in scripts/*.py charts/*.py; do
  python3 -m py_compile "$f" 2>/dev/null || { note "syntax $f" "FAIL"; syn=1; }
done
rm -rf scripts/__pycache__ charts/__pycache__
if [ "$syn" = 0 ]; then note "script syntax" "ok"; else fail=1; fi

# 3. manifest and charts are regenerable and unchanged
before=$(cat data/manifest.csv charts/*.svg | sha256sum)
gen=0
python3 scripts/build_manifest.py >/dev/null || { note "manifest" "FAIL"; fail=1; gen=1; }
python3 charts/make_charts.py >/dev/null || { note "charts" "FAIL"; fail=1; gen=1; }
after=$(cat data/manifest.csv charts/*.svg | sha256sum)
if [ "$gen" != 0 ]; then
  note "manifest+charts" "not checked, generator failed"
elif [ "$before" = "$after" ]; then
  note "manifest+charts" "ok, up to date"
else
  note "manifest+charts" "REGENERATED - commit the change"
  fail=1
fi

# 4. nothing host-specific slipped in.
# Patterns are split so this file never contains the literals it searches for,
# which also stops a sanitiser pass from rewriting the check itself.
u='pf'; u="${u}abi"
n='Fabi'; n="${n}szewski"
h='atomic-pc|agentsjump|claudejump|llamaproxy|gpuremote|lumo-pve'
pat="$u|/var/home/|$n|($h)"
pat="$pat"'|[A-Za-z0-9-]+\.lan\b'
# Full dotted quads only: a three-group form matches log timestamps like
# "1.10.991.337" in kv-startup-128k.log.
pat="$pat"'|\b(10(\.[0-9]{1,3}){3}|192\.168(\.[0-9]{1,3}){2}|172\.(1[6-9]|2[0-9]|3[01])(\.[0-9]{1,3}){2})\b'
pat="$pat"'|([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
pat="$pat"'|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
# Elapsed log clocks can look like private IPs (four dotted numeric groups).
# Strip only a leading clock in a .log hit, then rescan the rest of that line.
# An actual address later on the same line must still fail this check.
hits=$(grep -rInE "$pat" . --exclude-dir=.git --exclude=check.sh |
  sed -E 's/^([^:]+\.log:[0-9]+:)[0-9]+(\.[0-9]+){3} ([IWE]) /\1\3 /' |
  grep -E "$pat" || true)
if [ -n "$hits" ]; then echo "$hits"; note "no host identifiers" "FAIL"; fail=1
else note "no host identifiers" "ok"; fi

exit $fail
