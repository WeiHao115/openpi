from openpi.shared import download

REMOTE_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_base"


def _contains_files(path):
    return path.exists() and any(item.is_file() for item in path.rglob("*"))


cache_dir = download.get_cache_dir()
expected_dir = cache_dir / "openpi-assets" / "checkpoints" / "pi05_base"

checkpoint_dir = download.maybe_download(
    REMOTE_CHECKPOINT,
    force_download=expected_dir.exists() and not _contains_files(expected_dir),
    gs={"token": "anon"},
)

print(checkpoint_dir)
