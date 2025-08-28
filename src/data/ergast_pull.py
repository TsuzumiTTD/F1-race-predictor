import argparse, requests, pandas as pd, time, sys

BASE = "https://ergast.com/api/f1"

def fetch(endpoint, params=None):
    url = f"{BASE}/{endpoint}.json"
    p = {"limit": 1000}
    if params: p.update(params)
    r = requests.get(url, params=p, timeout=30)
    r.raise_for_status()
    return r.json()

def races_df(season):
    j = fetch(f"{season}")
    races = j["MRData"]["RaceTable"]["Races"]
    rows = []
    for r in races:
        rows.append({
            "season": int(season),
            "round": int(r["round"]),
            "raceName": r["raceName"],
            "date": r["date"],
            "circuitId": r["Circuit"]["circuitId"],
            "circuitName": r["Circuit"]["circuitName"],
            "country": r["Circuit"]["Location"].get("country"),
            "lat": r["Circuit"]["Location"].get("lat"),
            "long": r["Circuit"]["Location"].get("long"),
        })
    return pd.DataFrame(rows)

def results_df(season):
    j = fetch(f"{season}/results", {"limit": 2000})
    races = j["MRData"]["RaceTable"]["Races"]
    rows = []
    for r in races:
        for res in r["Results"]:
            rows.append({
                "season": int(season),
                "round": int(r["round"]),
                "raceName": r["raceName"],
                "driverId": res["Driver"]["driverId"],
                "constructorId": res["Constructor"]["constructorId"],
                "grid": int(res.get("grid", 0) or 0),
                "position": int(res["position"]) if res.get("position") and res["position"].isdigit() else None,
                "status": res.get("status"),
                "points": float(res.get("points", 0.0)),
            })
    return pd.DataFrame(rows)

def quali_df(season):
    j = fetch(f"{season}/qualifying", {"limit": 2000})
    races = j["MRData"]["RaceTable"]["Races"]
    rows = []
    def t2s(t):
        # convert "1:23.456" to seconds (83.456), return None if missing
        if not t: return None
        if ":" in t:
            m, s = t.split(":")
            try: return int(m)*60 + float(s)
            except: return None
        try: return float(t)
        except: return None

    for r in races:
        for q in r["QualifyingResults"]:
            rows.append({
                "season": int(season),
                "round": int(r["round"]),
                "driverId": q["Driver"]["driverId"],
                "constructorId": q["Constructor"]["constructorId"],
                "q1_s": t2s(q.get("Q1")),
                "q2_s": t2s(q.get("Q2")),
                "q3_s": t2s(q.get("Q3")),
                "grid": int(q.get("position", 0) or 0),
            })
    return pd.DataFrame(rows)

def main(start, end):
    all_r, all_res, all_q = [], [], []
    for y in range(start, end+1):
        print(f"Fetching {y}…", file=sys.stderr)
        all_r.append(races_df(y))
        all_res.append(results_df(y))
        all_q.append(quali_df(y))
        time.sleep(0.5)  # be polite to the API
    pd.concat(all_r, ignore_index=True).to_csv("data/raw/races.csv", index=False)
    pd.concat(all_res, ignore_index=True).to_csv("data/raw/results.csv", index=False)
    pd.concat(all_q, ignore_index=True).to_csv("data/raw/qualifying.csv", index=False)
    print("Saved to data/raw/*.csv", file=sys.stderr)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--end", type=int, default=2024)
    args = ap.parse_args()
    main(args.start, args.end)
