#!/usr/bin/env bash
# Install overnight-runner as a USER systemd service.
#
# It does NOT:
#   - enable linger
#   - alter system-wide services
#   - require root
#   - kill Ollama
#   - start a run
#
# It DOES:
#   - create ~/.config/systemd/user if needed
#   - copy service + timer from this repo
#   - run daemon-reload
#   - enable the TIMER (so the timer runs when activated)
#
# After installation:
#   - verify with:   systemctl --user status overnight-runner.timer
#   - trigger now:   systemctl --user start overnight-runner.service
#   - logs:           journalctl --user -u overnight-runner.service
#
# The timer is Persistent=false so a missed night does not retry on next boot.
# The user must EITHER stay logged in overnight OR explicitly enable linger:
#     loginctl enable-linger "$USER"

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SERVICE_SRC="$REPO_ROOT/packaging/systemd/overnight-runner.service"
TIMER_SRC="$REPO_ROOT/packaging/systemd/overnight-runner.timer"
DEST_DIR="$HOME/.config/systemd/user"

if [[ ! -f "$SERVICE_SRC" || ! -f "$TIMER_SRC" ]]; then
    echo "missing service or timer file in $REPO_ROOT/packaging/systemd" >&2
    exit 2
fi

mkdir -p "$DEST_DIR"
cp "$SERVICE_SRC" "$DEST_DIR/overnight-runner.service"
cp "$TIMER_SRC"  "$DEST_DIR/overnight-runner.timer"
chmod 0644 "$DEST_DIR/overnight-runner.service" "$DEST_DIR/overnight-runner.timer"

systemctl --user daemon-reload
systemctl --user enable overnight-runner.timer

cat <<EOF

Installed:
  $DEST_DIR/overnight-runner.service
  $DEST_DIR/overnight-runner.timer

Timer enabled: overnight-runner.timer
Persistent=false (missed nights will NOT retry on next boot).

To start NOW manually:
  systemctl --user start overnight-runner.service

To check status:
  systemctl --user status overnight-runner.timer
  systemctl --user list-timers overnight-runner.timer

To view logs:
  journalctl --user -u overnight-runner.service

If you want the timer to fire when you are NOT logged in, enable linger:
  loginctl enable-linger "$USER"
(This script deliberately does NOT do that for you.)

EOF
