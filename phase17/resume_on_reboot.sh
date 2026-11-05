#!/bin/bash
# Phase 17A reboot safety (crontab @reboot). Resumes ONLY the Foundation v4 run, and only if it was launched after
# the stability gate passed (LAUNCHED marker) and has not completed. Remove the crontab entry on 2026-11-11
# (see phase17/CLEANUP_NOTES.md).
cd "$(dirname "$0")/.." || exit 1
RUN=phase17/runs/foundation_v4
echo "$(date '+%F %T') reboot check" >> phase17/logs/reboot_resume.log
[ -e "$RUN/LAUNCHED" ] || { echo "  not launched; nothing to do" >> phase17/logs/reboot_resume.log; exit 0; }
[ -e "$RUN/COMPLETED" ] && { echo "  completed; nothing to do" >> phase17/logs/reboot_resume.log; exit 0; }
for i in $(seq 1 60); do nvidia-smi > /dev/null 2>&1 && break; sleep 10; done  # wait up to 10 min for the GPU driver
echo "  resuming" >> phase17/logs/reboot_resume.log
setsid nohup phase17/run_foundation_v4.sh > /dev/null 2>&1 < /dev/null &
