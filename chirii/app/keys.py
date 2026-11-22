"""
S5: the data key (CHIRII_DATA_KEY) and encrypted backups. Run from the repository root, with the app stopped for
rotate:

    python -m chirii.app.keys create          make a new key and write it into .env (never printed)
    python -m chirii.app.keys show            print the key, for copying into a password manager (only when you run it)
    python -m chirii.app.keys check           check that the key in .env opens the data
    python -m chirii.app.keys rotate          make a new key and re-encrypt every user's file with it
    python -m chirii.app.keys backup FOLDER   copy the encrypted files (not the key, not .env) to FOLDER

If the key is lost, the data cannot be recovered.

Rotation is safe to interrupt: every file is first re-encrypted next to the old one (memory.enc.new), then .env gets
the new key with the old one kept as CHIRII_DATA_KEY_PREVIOUS, then the new files are swapped in, then the previous
key is removed. Running rotate again after an interruption finishes the job.
"""

import argparse
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from chirii.app.config import ENV_FILE, Settings
from chirii.app.memory.vault import (CHECK_FILE, CHECK_MAGIC, CHECK_TEXT, DATA_FILE, Vault, atomic_write, check_key,
                                     derive_key, encrypt, new_key)

NEW = ".new"


# ---------------------------------------------------------------- .env, without ever printing a value

def read_env(path: Path) -> Dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                values[name.strip()] = value.strip()
    return values


def set_env(path: Path, name: str, value: Optional[str]) -> None:
    """Sets (or with value None, removes) one line of .env, keeping every other line as it was."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out, done = [], False
    for line in lines:
        if line.split("=", 1)[0].strip() == name and not line.lstrip().startswith("#"):
            if value is not None and not done:
                out.append(f"{name}={value}")
                done = True
            continue
        out.append(line)
    if value is not None and not done:
        out.append(f"{name}={value}")
    atomic_write(path, ("\n".join(out) + "\n").encode("utf-8"))


# ---------------------------------------------------------------- rotation

def user_folders(data_dir: Path) -> List[Path]:
    users = Path(data_dir) / "users"
    return sorted(p.parent for p in users.glob(f"*/{DATA_FILE}")) if users.exists() else []


def prepare_rotation(data_dir: Path, old_key: str, new_key_: str) -> int:
    """Step 1: re-encrypt every user's file and the check value next to the old ones. Nothing old is changed."""
    folders = user_folders(data_dir)
    for folder in folders:
        plain = Vault(folder, old_key).read()
        target = Vault(folder, new_key_)
        atomic_write(folder / (DATA_FILE + NEW), encrypt(target.key, plain, target.bound_to))
    atomic_write(Path(data_dir) / (CHECK_FILE + NEW), encrypt(derive_key(new_key_, "check"), CHECK_TEXT, "check", CHECK_MAGIC))
    return len(folders)


def finish_rotation(data_dir: Path) -> int:
    """Step 3: swap the new files in (each swap is atomic)."""
    swapped = 0
    for new in list(Path(data_dir).glob(f"users/*/{DATA_FILE}{NEW}")) + [Path(data_dir) / (CHECK_FILE + NEW)]:
        if new.exists():
            new.replace(new.with_name(new.name[: -len(NEW)]))
            swapped += 1
    return swapped


def rotate(env_path: Path, data_dir: Path) -> int:
    env = read_env(env_path)
    current, previous = env.get("CHIRII_DATA_KEY"), env.get("CHIRII_DATA_KEY_PREVIOUS")
    if previous:                                   # an interrupted rotation: .env already has the new key
        finish_rotation(data_dir)
        check_key(data_dir, current)
        set_env(env_path, "CHIRII_DATA_KEY_PREVIOUS", None)
        return len(user_folders(data_dir))
    if not current:
        raise SystemExit("No CHIRII_DATA_KEY in .env. Run: python -m chirii.app.keys create")
    for stale in Path(data_dir).glob(f"**/*{NEW}"):
        stale.unlink()                             # left from a rotation that stopped before .env changed
    check_key(data_dir, current)
    fresh = new_key()
    count = prepare_rotation(data_dir, current, fresh)
    set_env(env_path, "CHIRII_DATA_KEY_PREVIOUS", current)
    set_env(env_path, "CHIRII_DATA_KEY", fresh)
    finish_rotation(data_dir)
    check_key(data_dir, fresh)
    set_env(env_path, "CHIRII_DATA_KEY_PREVIOUS", None)
    return count


# ---------------------------------------------------------------- backup

def backup(data_dir: Path, destination: Path) -> Path:
    """Copies the encrypted files only. The key and .env are never copied: a backup is useless without the key."""
    data_dir, destination = Path(data_dir).resolve(), Path(destination).resolve()
    if destination == data_dir or data_dir in destination.parents:
        raise SystemExit("Choose a backup folder outside the data folder.")
    target = destination / f"chirii-backup-{datetime.now():%Y%m%d-%H%M%S}"
    files = [data_dir / CHECK_FILE] + [f / DATA_FILE for f in user_folders(data_dir)]
    for f in files:
        if f.exists():
            out = target / f.relative_to(data_dir)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
    return target


def app_is_running(port: int = 8000) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m chirii.app.keys", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "show", "check", "rotate"):
        sub.add_parser(name)
    sub.add_parser("backup").add_argument("folder")
    args = parser.parse_args(argv)
    env = read_env(ENV_FILE)
    data_dir = Settings().data_dir

    if args.command == "create":
        if env.get("CHIRII_DATA_KEY"):
            raise SystemExit("CHIRII_DATA_KEY is already set in .env; nothing changed. (A new key would lock out the data.)")
        set_env(ENV_FILE, "CHIRII_DATA_KEY", new_key())
        print(f"A new CHIRII_DATA_KEY was written to {ENV_FILE} (not shown).")
        print("Save a copy in your password manager now:  python -m chirii.app.keys show")
    elif args.command == "show":
        if not env.get("CHIRII_DATA_KEY"):
            raise SystemExit("No CHIRII_DATA_KEY in .env.")
        print("CHIRII_DATA_KEY (keep it secret; if it is lost, the data cannot be recovered):")
        print(env["CHIRII_DATA_KEY"])
    elif args.command == "check":
        print({"created": "Key check value created.", "ok": "The key opens the data."}[check_key(data_dir, env.get("CHIRII_DATA_KEY", ""))])
    elif args.command == "rotate":
        if app_is_running():
            raise SystemExit("Stop the app first (something is answering on port 8000).")
        print(f"Re-encrypted {rotate(ENV_FILE, data_dir)} user file(s) with a new key (not shown). "
              "Save the new key in your password manager:  python -m chirii.app.keys show")
    elif args.command == "backup":
        print(f"Encrypted files copied to {backup(data_dir, Path(args.folder))}. The key is not in the backup.")


if __name__ == "__main__":
    main(sys.argv[1:])
