
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--history",required=True)
    ap.add_argument("--output",default="smart_scan_audit.xlsx")
    args=ap.parse_args()
    df=pd.read_csv(args.history)
    with pd.ExcelWriter(args.output,engine="openpyxl") as w:
        df.to_excel(w,index=False,sheet_name="Scan History")
        if "policy" in df:
            s=df.groupby("policy").agg(
                scans=("policy","size"),
                detections=("observation_hit","sum"),
                mean_snr_db=("mean_snr_db","mean"),
                mean_belief=("belief_probability","mean")
            ).reset_index()
            s["detection_rate"]=s.detections/s.scans
            s.to_excel(w,index=False,sheet_name="Summary")
        if "selected_center_mhz" in df:
            b=df.groupby(["policy","selected_center_mhz"]).agg(
                scans=("selected_center_mhz","size"),
                detections=("observation_hit","sum")
            ).reset_index()
            b["hit_rate"]=b.detections/b.scans
            b.to_excel(w,index=False,sheet_name="Band Statistics")
    print("Saved:",args.output)
if __name__=="__main__": main()
