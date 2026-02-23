"""
NRC-VAD Lexicon auto-download utility.
The NRC-VAD lexicon is free for non-commercial research and educational use
but may not be redistributed. This module fetches it directly from the
official source (saifmohammad.com) on first use.
Citation: Mohammad, Saif M. "Obtaining Reliable Human Ratings of Valence,
Arousal, and Dominance for 20,000 English Words." ACL 2018.
Homepage: http://saifmohammad.com/WebPages/nrc-vad.html
"""
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

NRC_VAD_URL = "http://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip"
NRC_VAD_DEFAULT_PATH = "data/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"


def ensure_nrc_lexicon(lexicon_path: str = NRC_VAD_DEFAULT_PATH) -> None:
    """
    Ensure the NRC-VAD lexicon exists at lexicon_path.
    If not present, downloads and extracts it from the official NRC source.
    Raises RuntimeError if the download fails or the file is still missing
    after extraction.
    """
    target = Path(lexicon_path)
    if target.exists():
        return

    print(
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  NRC-VAD Lexicon not found — downloading from official source\n"
        "\n"
        "  Created by Saif Mohammad at the National Research Council Canada.\n"
        "  Free for non-commercial research and educational use.\n"
        "  Homepage: http://saifmohammad.com/WebPages/nrc-vad.html\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "NRC-VAD-Lexicon-v2.1.zip"
            logger.info("Downloading NRC-VAD lexicon from %s", NRC_VAD_URL)
            print(f"  Downloading {NRC_VAD_URL} ...")
            req = urllib.request.Request(
                NRC_VAD_URL,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            )
            with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
                out_file.write(response.read())
            print("  Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)
            # Find the .txt file anywhere in the extracted tree
            matches = list(Path(tmp_dir).rglob("NRC-VAD-Lexicon-v2.1.txt"))
            if not matches:
                raise RuntimeError(
                    "NRC-VAD-Lexicon-v2.1.txt not found in downloaded archive. "
                    "The archive layout may have changed — please download manually "
                    "from http://saifmohammad.com/WebPages/nrc-vad.html"
                )
            shutil.copy2(matches[0], target)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to download NRC-VAD lexicon: {e}\n"
            "Please download it manually from "
            "http://saifmohammad.com/WebPages/nrc-vad.html "
            f"and place NRC-VAD-Lexicon-v2.1.txt at: {target}"
        ) from e

    if not target.exists():
        raise RuntimeError(
            f"NRC-VAD lexicon download appeared to succeed but file is missing at {target}."
        )

    print(f"  ✓ NRC-VAD lexicon saved to {target}\n")
    logger.info("NRC-VAD lexicon downloaded to %s", target)