"""Downloads "round 2": whole-vine photos from the Flavescence dorée dataset (Mendeley, CC BY 4.0),
ONLY the ones not used to train the diseased-leaf model (by photo number, per campaign).

Run (from the model folder):
  python tools/download_round2.py                  # Cabernet Sauvignon 2020 only (~144 photos, ~200 MB)
  python tools/download_round2.py --campaigns all  # all campaigns (~290 photos, ~380 MB)
"""
import argparse
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets/3dr9r3w3jn"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# symptom_scale_box: the photos used to train the diseased-leaf model (campaign -> folders)
TRAINING = {
    "CS20": ["cc48eedb-c553-40cc-8d10-409cd5115763"],
    "UB20": ["d805e05d-e696-4fc5-9256-6eef0f22b529", "74f8f93f-762d-4f23-b28b-b0c24a9bd66d"],
    "M21": ["fbdd115d-f7ed-4494-a14e-71d58d246165"],
    "SB21": ["8c069f41-c13b-464a-a1c4-c1b0a962fcaa"],
}
# image_scale: the "diagnostic" photos, by category (campaign -> category -> folder)
DIAGNOSTIC = {
    "CS20": {"CONF": "49b09f46-abef-4219-a2dc-31bbeedb0ae1", "CONF+": "146d3085-26af-491a-a323-6a29b7b3e15f",
             "ESCA": "b8bdb377-7c3b-4db7-927f-1564d9c58378", "FD": "a4d340c0-bae4-4d64-81e8-f94cb3d40e5f"},
    "M21": {"CONF": "bdb3164d-e786-4491-ac6a-20ec685a5a12", "CONF+": "fded502f-4815-47c0-8d8a-d4f1968bf47b",
            "ESCA": "4e243631-e33f-4a27-a693-bfa939910b02", "FD": "1c32c600-13e1-4abf-80f9-e78bfeac12aa"},
    "SB21": {"CONF": "3de51478-024e-45c4-b9db-1810d49f9d99", "CONF+": "cb494e4c-e8e8-41e2-9aca-fff75a7f52f4",
             "ESCA": "839a68f8-d99c-4e50-a0d6-75da725b77a2", "FD": "1e1336a9-a289-4db0-86bc-842dfec98395"},
    "UB20": {"CONF": "f68f0595-c83a-45df-a1a5-1b88fb9ace0f", "CONF+": "534846a6-8d0a-44e0-aeb8-6c422c600943",
             "ESCA": "de5e857b-bbae-4814-bf59-5da7d655605b", "FD": "6dfc42bc-ec34-4044-b97a-f3799d9de922"},
}


def get(url, as_json=True):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r:
        data = r.read()
    return json.loads(data) if as_json else data


def list_folder(folder_id):
    return get(f"{API}/files?folder_id={folder_id}&version=2")


def photo_number(name):
    """"im_00031 (2).jpg" -> 31 ; "im_SB21_00001.jpg" -> 1"""
    numbers = re.findall(r"\d{3,}", name.split("(")[0])
    return int(numbers[-1]) if numbers else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaigns", default="CS20", help="CS20 (default), a list e.g. CS20,SB21, or \"all\"")
    p.add_argument("--output", default="data/round2")
    p.add_argument("--list-only", action="store_true", help="show how many photos and MB, without downloading")
    args = p.parse_args()
    campaigns = list(DIAGNOSTIC) if args.campaigns == "all" else args.campaigns.split(",")

    jobs = []
    for camp in campaigns:
        seen = {photo_number(f["filename"]) for fid in TRAINING[camp] for f in list_folder(fid)
                if f["filename"].lower().endswith(".jpg")}
        for category, fid in DIAGNOSTIC[camp].items():
            for f in list_folder(fid):
                n = photo_number(f["filename"])
                if not f["filename"].lower().endswith(".jpg") or n is None or n in seen:
                    continue
                dup = re.search(r"\((\d+)\)", f["filename"])
                name = f"{camp}_{n:05d}{'_dup' + dup.group(1) if dup else ''}_{category.replace('+', 'plus')}.jpg"
                jobs.append((Path(args.output) / camp / name, f["content_details"]["download_url"], f["size"]))

    total_mb = sum(j[2] for j in jobs) / 1e6
    print(f"{len(jobs)} photos unseen in training, ~{total_mb:.0f} MB -> {args.output}")
    if args.list_only:
        return

    def download(job):
        path, url, _ = job
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(get(url, as_json=False))

    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(download, jobs), 1):
            if i % 20 == 0 or i == len(jobs):
                print(f"Downloaded {i}/{len(jobs)} ({i / len(jobs):.0%})", flush=True)


if __name__ == "__main__":
    main()
