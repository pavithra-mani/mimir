"""
Get this eval's Video-MME data onto your machine.

By default this does NOT download the ~46GB of chunks 1-7 video files -
nothing in this eval (generate_summaries.py, generate_references.py) reads
the actual video content, since transcription is mocked with subtitles and
there's no keyframe/visual metric here. Only subtitle.zip (7.5MB, covers the
whole dataset) and the metadata parquet (405KB) are needed, filtered down to
the 317 video IDs in chunks 1-7 (chunk_1-7_video_ids.txt, already computed
via a cheap remote-zip listing - see the README for how that manifest was
made). Pass --with-videos if you actually need the video files for something
outside this eval's scope.

Setup:
    pip install huggingface_hub

Usage:
    python eval/download_chunks.py --output-dir eval/data
    python eval/download_chunks.py --output-dir eval/data --with-videos
"""

import argparse
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "lmms-eval/Video-MME"
REPO_TYPE = "dataset"
METADATA_PATH = "videomme/test-00000-of-00001.parquet"
MANIFEST = Path(__file__).resolve().parent / "chunk_1-7_video_ids.txt"


def download_subtitles_and_metadata(output_dir: Path) -> tuple[Path, Path]:
    print("Fetching subtitle.zip (7.5MB, whole-dataset) + metadata parquet ...")
    subtitle_zip = Path(hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename="subtitle.zip", local_dir=output_dir))
    metadata = Path(hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=METADATA_PATH, local_dir=output_dir))
    return subtitle_zip, metadata


def extract_filtered_subtitles(subtitle_zip: Path, manifest: Path, dest_dir: Path) -> tuple[int, int]:
    video_ids = [l.strip() for l in manifest.read_text().splitlines() if l.strip()]

    extract_tmp = dest_dir / "_subtitle_zip_tmp"
    extract_tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(subtitle_zip) as zf:
        zf.extractall(extract_tmp)

    # The zip's srt files live under a "subtitle/" prefix
    src_dir = extract_tmp / "subtitle"
    if not src_dir.exists():
        src_dir = extract_tmp  # fall back in case the layout differs

    dest_dir.mkdir(parents=True, exist_ok=True)
    found, missing = 0, 0
    for vid in video_ids:
        src = src_dir / f"{vid}.srt"
        if src.exists():
            shutil.copyfile(src, dest_dir / f"{vid}.srt")
            found += 1
        else:
            missing += 1

    shutil.rmtree(extract_tmp)
    return found, missing


def download_video_chunks(output_dir: Path, chunks: list[int]):
    print(f"Fetching {len(chunks)} video chunk zips (~5GB each - this is the slow, heavy part) ...")
    for n in chunks:
        filename = f"videos_chunked_{n:02d}.zip"
        print(f"  {filename} ...")
        path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=filename, local_dir=output_dir)
        print(f"    -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="Video-ID list to filter subtitles against")
    ap.add_argument("--with-videos", action="store_true", help="Also download the actual chunks 1-7 video zips (~46GB, not needed for this eval)")
    ap.add_argument("--chunks", type=int, nargs="+", default=list(range(1, 8)), help="Chunk numbers, only used with --with-videos")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    subtitle_zip, metadata = download_subtitles_and_metadata(args.output_dir)

    subs_out = args.output_dir / "subtitles"
    found, missing = extract_filtered_subtitles(subtitle_zip, args.manifest, subs_out)
    print(f"\nExtracted {found} subtitle files to {subs_out} ({missing} of the manifest's videos have no subtitles on YouTube - expected, not an error)")

    if args.with_videos:
        download_video_chunks(args.output_dir, args.chunks)

    print(f"\nDone. Point generate_summaries.py --subs-dir at: {subs_out}")


if __name__ == "__main__":
    main()
