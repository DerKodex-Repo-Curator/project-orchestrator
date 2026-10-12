#!/usr/bin/env bash
# Build `nexus-tools` (the files, shell, web and search tools of a native session, and the
# browser) at the nexus revision Cargo.toml pins, with https and Obscura embedded
# (`--features tls,obscura`), into <root>/bin.
#
# Why here: a native session has no Read/Bash/WebFetch unless this executable is next to the
# server, on the PATH, or named by NEXUS_TOOLS_PATH (src/chat/config.rs, detect_nexus_tools_path).
# Every release channel ships it NEXT TO the `orchestrator` executable (archives, Homebrew, .deb,
# .rpm, Docker, desktop app); the CI builds it the same way (.github/actions/nexus-fakes), so the
# pinned revision is known to build before a release needs it.
#
# `obscura` (decision 1e9d7501): the keyless WebSearch engine and the browser run inside
# nexus-tools, nothing to install next to it. It links V8, a prebuilt static library that the
# `v8` build script would download unverified; the nexus repository's scripts/rusty-v8-cache.sh,
# at the same pinned revision, checks its SHA-256 and puts it in rusty_v8's cache first, for the
# host and the target. The build always names its --target (the host's by default): without it,
# the host and target copies of `v8` share one directory and can overwrite each other's library.
#
# The revision is the one of `nexus-claude` in Cargo.toml; every nexus dependency there must
# carry the same one (a server built against one revision ships the tools of that revision).
#
# NEXUS_TOOLS_FEATURES overrides the features (e.g. `tls` for a target without a prebuilt V8, or
# the macOS 10.15 "legacy" build, which V8's prebuilt library does not promise to support).
#
# Usage: scripts/build-nexus-tools.sh <root> [cargo install args...]   (e.g. --target <triple>)
set -euo pipefail

root="${1:?usage: $0 <root> [cargo install args...]}"
shift

manifest="${CARGO_MANIFEST:-Cargo.toml}"
revs="$(sed -n 's/^nexus-[a-z-]* = .*github.com\/this-rs\/nexus.*rev = "\([0-9a-f]\{40\}\)".*/\1/p' "$manifest" | sort -u)"
count="$(printf '%s\n' "$revs" | grep -c . || true)"
if [ "$count" -ne 1 ]; then
  echo "::error::expected exactly one nexus revision in $manifest, found $count: $revs" >&2
  exit 1
fi
rev="$revs"
features="${NEXUS_TOOLS_FEATURES:-tls,obscura}"

host="$(rustc -vV | sed -n 's/^host: //p')"
target=""
prev=""
for arg in "$@"; do
  case "$arg" in
    --target=*) target="${arg#--target=}" ;;
  esac
  [ "$prev" = "--target" ] && target="$arg"
  prev="$arg"
done
extra=()
if [ -z "$target" ]; then
  target="$host"
  extra=(--target "$target")
fi

case ",$features," in
  *,obscura,*)
    # The checksum table and its version pin come from the nexus revision being built.
    seed="$(mktemp -d)"
    trap 'rm -rf "$seed"' EXIT
    git -C "$seed" init -q
    git -C "$seed" fetch -q --depth 1 https://github.com/this-rs/nexus.git "$rev"
    git -C "$seed" checkout -q FETCH_HEAD -- scripts/rusty-v8-cache.sh obscura/crates/obscura-js/Cargo.toml
    targets=("$target")
    [ "$host" != "$target" ] && targets+=("$host")
    bash "$seed/scripts/rusty-v8-cache.sh" "${targets[@]}"
    ;;
esac

echo "nexus-tools at this-rs/nexus@$rev (features: $features, target: $target) -> $root/bin"
cargo install --git https://github.com/this-rs/nexus.git --rev "$rev" \
  --features "$features" --bin nexus-tools --root "$root" "${extra[@]+"${extra[@]}"}" "$@" nexus-tools
