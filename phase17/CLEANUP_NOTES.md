# Phase 17A — cleanup notes (do on 2026-11-11)

## 1. Remove the reboot crontab entry (required)

Installed on 2026-11-04 for the Foundation v4 run only. It runs `phase17/resume_on_reboot.sh` at boot, which resumes
the run only if `phase17/runs/foundation_v4/LAUNCHED` exists and `COMPLETED` does not; otherwise it does nothing.

The entry (user `crontab`, not system-wide):

```
@reboot ~/chirag_work/minigpt/phase17/resume_on_reboot.sh >> ~/chirag_work/minigpt/phase17/logs/reboot_resume.log 2>&1 # MINIGPT-PHASE17A remove on 2026-11-11 (phase17/CLEANUP_NOTES.md)
```

(The installed line uses the absolute home path.) To remove only this entry and keep anything else:

```
crontab -l | grep -v "MINIGPT-PHASE17A" | crontab -
crontab -l      # check: the MINIGPT-PHASE17A line is gone
```

There was no crontab before this entry, so after removal `crontab -l` should say "no crontab".

## 2. Large local files (optional, after the results are backed up)

| Path | Size on 2026-11-04 | What it is |
| :--- | ---: | :--- |
| `phase17/data/` | 19 GB | Phase 17A text and token shards, `train_3b.bin` (hashes in `phase17/manifests/`) |
| `phase17/runs/` | grows during training | checkpoints (resumable `checkpoint.pt`) and weights snapshots every 2,000 steps |
| `~/chirag_work/hf_cache/` | 5.3 GB | Hugging Face models and benchmark datasets (Phases 16 and 18) |
| `~/chirag_work/pylibs/phase16`, `phase18` | 0.1 GB, 0.3 GB | transformers / lm-eval installed outside conda |
| `~/chirag_work/tmp/` | 1.5 GB | scratch: publishing clones, patches, Phase 18 full result backup |

Nothing here is deleted automatically.
