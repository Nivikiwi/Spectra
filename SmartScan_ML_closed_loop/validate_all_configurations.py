
import argparse
from pathlib import Path
import re
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True)
    args=ap.parse_args()
    files=sorted(Path(args.data_dir).glob("config_*.h5"),key=lambda p:int(re.search(r"config_(\d+)",p.name).group(1)))
    ids=[int(re.search(r"config_(\d+)",p.name).group(1)) for p in files]
    print(f"Configuration files found: {len(files)}")
    print("IDs:",ids)
    if not files: raise SystemExit("No config_*.h5 files found.")
if __name__=="__main__": main()
