import os, time, requests
from pathlib import Path
from tqdm import tqdm

# 1) Config from env
ACCESS_KEY   = os.getenv("UNSPLASH_ACCESS_KEY")
assert ACCESS_KEY, "Set UNSPLASH_ACCESS_KEY first!"
ROOT = Path("dataset_raw/unsplash_raw")
ROOT.mkdir(parents=True, exist_ok=True)

TOTAL = 1000
BATCH = 30  # max per request
headers = {"Authorization": f"Client-ID {ACCESS_KEY}"}

downloaded = 0
pbar = tqdm(total=TOTAL)

while downloaded < TOTAL:
    n = min(BATCH, TOTAL - downloaded)
    resp = requests.get(
        "https://api.unsplash.com/photos/random",
        params={"count": n, "orientation": "landscape"},
        headers=headers
    )
    resp.raise_for_status()
    for img in resp.json():
        # get a reasonably sized JPG (e.g. w=1024)
        url = img["urls"]["raw"] + "&w=1024"
        data = requests.get(url).content
        fname = ROOT / f"img{downloaded+1:04d}.jpg"
        with open(fname, "wb") as f:
            f.write(data)
        downloaded += 1
        pbar.update(1)
    time.sleep(1)  # respect rate limits

pbar.close()
print("Downloaded 1 000 images to", ROOT)
