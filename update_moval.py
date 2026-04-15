import argparse
import subprocess
import sys
from typing import List, Optional


def run_git(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def ensure_git_repository() -> None:
    try:
        run_git(["rev-parse", "--show-toplevel"])
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or "Not a git repository."
        print(stderr)
        sys.exit(1)


def get_worktree_changes() -> List[str]:
    result = run_git(["status", "--porcelain"])
    return [line for line in result.stdout.splitlines() if line.strip()]


def confirm_dirty_worktree_action(changes: List[str]) -> bool:
    print("Working tree has uncommitted changes:")
    for line in changes[:10]:
        print(f"  {line}")
    if len(changes) > 10:
        print(f"  ... and {len(changes) - 10} more")
    print("")
    print("Choose how to proceed:")
    print("  [K]eep current changes and cancel update")
    print("  [F]orce update and discard local changes")

    while True:
        choice = input("Enter K or F [K]: ").strip().lower()
        if choice in ("", "k", "keep"):
            print("Update cancelled. Your local changes were kept.")
            return False
        if choice in ("f", "force"):
            return True
        print("Please enter K or F.")


def discard_local_changes() -> None:
    print("Discarding local changes...")
    run_git(["reset", "--hard", "HEAD"])
    run_git(["clean", "-fd"])


def handle_dirty_worktree() -> None:
    changes = get_worktree_changes()
    if not changes:
        return
    if not confirm_dirty_worktree_action(changes):
        sys.exit(0)
    discard_local_changes()


def fetch_tags() -> None:
    print("Fetching latest tags from origin...")
    run_git(["fetch", "origin", "--tags"])


def list_tags() -> List[str]:
    result = run_git(["tag", "--sort=-version:refname"])
    return [tag.strip() for tag in result.stdout.splitlines() if tag.strip()]


def resolve_target_tag(requested_version: Optional[str]) -> str:
    tags = list_tags()
    if not tags:
        print("No git tags found in this repository.")
        sys.exit(1)

    if requested_version:
        if requested_version not in tags:
            print(f"Tag '{requested_version}' was not found.")
            sys.exit(1)
        return requested_version

    return tags[0]


def checkout_tag(tag: str) -> None:
    print(f"Checking out {tag}...")
    run_git(["checkout", "-f", tag])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update MovAl to the latest release tag or a specific tag."
    )
    parser.add_argument(
        "version",
        nargs="?",
        help="Target release tag (example: v1.1.3). If omitted, the latest tag is used.",
    )
    args = parser.parse_args()

    ensure_git_repository()
    handle_dirty_worktree()
    fetch_tags()
    target_tag = resolve_target_tag(args.version)
    checkout_tag(target_tag)
    print(f"MovAl is now set to {target_tag}.")


if __name__ == "__main__":
    main()
