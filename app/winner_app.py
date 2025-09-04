import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import requests
from streamlit import column_config
import math
from typing import Dict, Iterable
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
from itertools import combinations
import unicodedata, re



# ISO country flags (emoji)
CIRCUIT_FLAGS = {
    "Australia": "🇦🇺",
    "Austria": "🇦🇹",
    "Azerbaijan": "🇦🇿",
    "Bahrain": "🇧🇭",
    "Belgium": "🇧🇪",
    "Brazil": "🇧🇷",
    "Canada": "🇨🇦",
    "China": "🇨🇳",
    "France": "🇫🇷",
    "Great Britain": "🇬🇧", "United Kingdom": "🇬🇧",
    "Hungary": "🇭🇺",
    "Italy": "🇮🇹",
    "Japan": "🇯🇵",
    "Mexico": "🇲🇽",
    "Monaco": "🇲🇨",
    "Netherlands": "🇳🇱",
    "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦",
    "Singapore": "🇸🇬",
    "Spain": "🇪🇸",
    "United States": "🇺🇸", "USA": "🇺🇸",
    "Abu Dhabi": "🇦🇪", "UAE": "🇦🇪", "United Arab Emirates": "🇦🇪",
}

NICE_LABELS = {
    "grid": "Starting grid",
    "grid_quali": "Qualifying position",
    "grid_sq": "Grid²",
    "grid_log": "log(1+grid)",
    "front_row": "Front row (≤ P2)",
    "start_penalty": "Start penalty (grid − quali)",

    "drv_win_rate_career": "Driver win rate (career)",
    "drv_podium_rate_career": "Driver podium rate (career)",
    "drv_pts_per_start_career": "Driver points/start (career)",
    "drv_years_exp_so_far": "Driver years of experience",
    "drv_starts_cum": "Career starts",
    "drv_wins_cum": "Career wins",
    "drv_podiums_cum": "Career podiums",

    "drv_track_win_rate": "Track win rate (driver)",
    "drv_track_podium_rate": "Track podium rate (driver)",
    "drv_track_starts_cum": "Track starts (driver)",
    "drv_track_wins_cum": "Track wins (driver)",
    "drv_track_podiums_cum": "Track podiums (driver)",

    "wx_is_wet": "Wet race",
    "wx_is_hot": "Hot race (≥30°C)",
    "wx_temp_mean_c": "Mean temp (°C)",
    "wx_temp_max_c": "Max temp (°C)",
    "wx_precip_mm": "Precip (mm)",
    "wx_windspeed_max_kmh": "Wind max (km/h)",
    # add others you use…
}

FETCH_LABEL = "Fetch latest qualifying (OpenF1)"
OPENF1_BASE = "https://api.openf1.org/v1"
DEFAULT_POS_POINTS = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}



st.set_page_config(page_title="F1 Winner Predictor", page_icon="🏁", layout="centered")

# ---- Cache helpers  ----
@st.cache_data
def load_csv(path):
    """Cached CSV reader with column de-dup & whitespace trim."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        # optional: surface once so you know what was removed
        dups = list(df.columns[df.columns.duplicated()])
        st.warning(f"Removed duplicate columns from {Path(path).name}: {dups}")
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


@st.cache_resource
def train_lr(X_df, y_ser):
    """Cached LogisticRegression training."""
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_df.values, y_ser.values)
    return clf


# --- NEW: local Kaggle CSV loader ---
def _norm_ids(df):
    m = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "raceid" and "raceId" not in df.columns: m[c] = "raceId"
        if lc == "driverid" and "driverId" not in df.columns: m[c] = "driverId"
        if lc == "year" and "season" not in df.columns:      m[c] = "season"
    return df.rename(columns=m)

def load_local_quali(season: int, rnd: int) -> pd.DataFrame:
    """
    Returns columns: driver, code, grid_quali, grid
    Data source: data/raw/{races,qualifying,drivers}.csv (from Kaggle)
    """
    raw = Path("data/raw")
    races_p = raw / "races.csv"
    qual_p  = raw / "qualifying.csv"
    drv_p   = raw / "drivers.csv"
    for pth in (races_p, qual_p, drv_p):
        if not pth.exists():
            raise FileNotFoundError(f"Missing required file: {pth}")

    races = _norm_ids(load_csv(races_p))
    qual  = _norm_ids(load_csv(qual_p))
    drv   = _norm_ids(load_csv(drv_p))

    race_row = races[(races["season"] == season) & (races["round"] == rnd)]
    if race_row.empty:
        return pd.DataFrame(columns=["driver","code","grid_quali","grid"])
    race_id = int(race_row.iloc[0]["raceId"])

    q = qual[qual["raceId"] == race_id].copy()
    if q.empty:
        return pd.DataFrame(columns=["driver","code","grid_quali","grid"])

    # handle driver names
    lower = {c.lower(): c for c in drv.columns}
    if "forename" not in lower and "givenname" in lower:
        drv = drv.rename(columns={lower["givenname"]: "forename"})
    if "surname" not in lower and "familyname" in lower:
        drv = drv.rename(columns={lower["familyname"]: "surname"})
    if "code" not in drv.columns:
        drv["code"] = drv.get("driverRef", drv.get("surname", ""))

    d_keep = drv[["driverId","code","forename","surname"]].copy()
    q = q.merge(d_keep, on="driverId", how="left")

    q["driver"] = q["forename"].fillna("") + " " + q["surname"].fillna("")
    q["grid_quali"] = pd.to_numeric(q["position"], errors="coerce")
    q = q.dropna(subset=["grid_quali"]).copy()
    q["grid"] = q["grid_quali"]

    return q[["driver","code","grid_quali","grid"]].sort_values("grid_quali").reset_index(drop=True)

# --- Driver mapping helpers (code/name -> driverId) ---
def load_driver_map():
    drv_p = Path("data/raw/drivers.csv")
    if not drv_p.exists():
        return None, None  # no mapping available
    drv = pd.read_csv(drv_p)
    # normalize id/name columns
    lower = {c.lower(): c for c in drv.columns}
    def col(*names):
        for n in names:
            if n in lower: return lower[n]
        return None
    idc   = col("driverid")
    codec = col("code")
    fnc   = col("forename","givenname")
    snc   = col("surname","familyname")
    if idc is None: return None, None
    if codec not in drv.columns: drv["code"] = drv.get(col("driverref"), drv.get(snc, ""))
    if fnc is None: drv["forename"] = ""
    if snc is None: drv["surname"]  = ""
    drv["full_name"] = (drv.get("forename","").astype(str) + " " + drv.get("surname","").astype(str)).str.strip()
    by_code = {str(c).upper(): int(i) for i, c in zip(drv[idc], drv.get("code","")) if pd.notna(c)}
    by_name = {str(n).lower(): int(i) for i, n in zip(drv[idc], drv["full_name"]) if pd.notna(n)}
    return by_code, by_name

BY_CODE, BY_NAME = load_driver_map()

# --- Circuit mapping (name/ref -> circuitId) ---
def load_circuits():
    cp = Path("data/raw/circuits.csv")
    if not cp.exists():
        return None, None, None
    c = pd.read_csv(cp)
    # normalize id/ref columns
    lower = {cname.lower(): cname for cname in c.columns}
    def col(*names):
        for n in names:
            if n in lower: return lower[n]
        return None
    cid = col("circuitid")
    cref = col("circuitref")
    name = col("name")
    if cid is None: 
        return None, None, None
    # string keys for UI
    c["__display__"] = c[name].fillna(c.get(cref, "")).astype(str) if name else c.get(cref, "").astype(str)
    by_display = {row["__display__"]: int(row[cid]) for _, row in c.iterrows() if pd.notna(row[cid])}
    by_ref     = {str(row.get(cref,"")).lower(): int(row[cid]) for _, row in c.iterrows() if pd.notna(row.get(cref))}
    return c, by_display, by_ref

CIRCUITS_DF, CIRCUIT_BY_NAME, CIRCUIT_BY_REF = load_circuits()

# --- Season → circuit list (round → circuit name/id) ---
def _norm_races_circuits():
    races_p = Path("data/raw/races.csv")
    circ_p  = Path("data/raw/circuits.csv")
    if not (races_p.exists() and circ_p.exists()):
        return None

    races_df = load_csv(races_p).copy()
    circ_df  = load_csv(circ_p).copy()

    # normalize columns
    # year -> season
    if "season" not in races_df.columns and "year" in races_df.columns:
        races_df = races_df.rename(columns={"year": "season"})
    # circuitid -> circuitId
    if "circuitId" not in races_df.columns:
        for c in races_df.columns:
            if c.lower() == "circuitid":
                races_df = races_df.rename(columns={c: "circuitId"})
                break
    if "circuitId" not in circ_df.columns:
        for c in circ_df.columns:
            if c.lower() == "circuitid":
                circ_df = circ_df.rename(columns={c: "circuitId"})
                break
    # attach circuit name
    name_col = None
    for cand in ["name", "circuitName"]:
        if cand in circ_df.columns:
            name_col = cand; break
    if name_col is None:
        circ_df["__circuit_name__"] = circ_df.get("circuitRef", "").astype(str)
        name_col = "__circuit_name__"

    out = races_df.merge(
        circ_df[["circuitId", name_col]].rename(columns={name_col: "circuitName"}),
        on="circuitId", how="left"
    )
    # keep only essentials
    keep = [c for c in ["season","round","raceId","circuitId","circuitName","name","date"] if c in out.columns]
    out = out[keep].copy()
    # sorting for UI
    sort_keys = []
    if {"season","round"}.issubset(out.columns): sort_keys = ["season","round"]
    elif "date" in out.columns:                  sort_keys = ["season","date"]
    if sort_keys:
        out = out.sort_values(sort_keys)
    return out

def build_season_circuit_index():
    rc = _norm_races_circuits()
    if rc is None or rc.empty:
        return {}
    idx = {}
    for season, grp in rc.groupby("season"):
        items = []
        for _, r in grp.iterrows():
            items.append({
                "round": int(r["round"]) if "round" in r and pd.notna(r["round"]) else None,
                "raceId": int(r["raceId"]) if "raceId" in r and pd.notna(r["raceId"]) else None,
                "circuitId": int(r["circuitId"]) if pd.notna(r["circuitId"]) else None,
                "label": f"R{int(r['round']):02d} — {str(r.get('circuitName') or r.get('name') or 'Unknown')}"
            })
        idx[int(season)] = items
    return idx

def _pick_value(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _best_driver_name(rec):
    # Prefer full/canonical names when OpenF1 provides them
    name = _pick_value(rec, "driver_name", "broadcast_name")
    if not name:
        first = _pick_value(rec, "first_name", default="")
        last  = _pick_value(rec, "last_name", default="")
        name = f"{first} {last}".strip()
    return name or None

def fetch_quali_openf1(season: int, round_num: int) -> pd.DataFrame:
    """
    Return DataFrame: driver (full name), code (3-letter), grid_quali, grid
    """
    try:
        # 1) Season+Round -> meeting_key
        m_resp = requests.get(f"{OPENF1_BASE}/meetings", params={"year": season}, timeout=25)
        m_resp.raise_for_status()
        meetings = m_resp.json()
        mtg = next((m for m in meetings if str(_pick_value(m, "round")) == str(round_num)), None)
        if not mtg:
            return pd.DataFrame(columns=["driver","code","grid_quali","grid"])
        meeting_key = _pick_value(mtg, "meeting_key", "key")

        # 2) Sessions for meeting -> pick Qualifying (or Sprint Shootout) and Race
        s_resp = requests.get(f"{OPENF1_BASE}/sessions", params={"meeting_key": meeting_key}, timeout=25)
        s_resp.raise_for_status()
        sessions = s_resp.json()

        def _is_quali(s):
            name = str(_pick_value(s, "session_name", "name", default="")).lower()
            typ  = str(_pick_value(s, "session_type", "type", default="")).lower()
            return ("qualifying" in name) or ("qualifying" in typ) or ("sprint shootout" in name) or ("sprint_shootout" in typ)

        quali = next((s for s in sessions if _is_quali(s)), None)
        race  = next((s for s in sessions if str(_pick_value(s, "session_name", "name", default="")).lower() == "race"), None)
        if not quali:
            return pd.DataFrame(columns=["driver","code","grid_quali","grid"])

        quali_key = _pick_value(quali, "session_key", "key")

        # 3) Pull QUALI classification
        q_resp = requests.get(f"{OPENF1_BASE}/session_result", params={"session_key": quali_key}, timeout=25)
        q_resp.raise_for_status()
        qres = q_resp.json()

        # 3a) Get driver dictionary for this session to map numbers -> names/codes
        d_resp = requests.get(f"{OPENF1_BASE}/drivers", params={"session_key": quali_key}, timeout=25)
        drivers = {}
        three_letter = {}
        teams = {}
        try:
            d_resp.raise_for_status()
            djson = d_resp.json()
            for d in djson:
                num = str(_pick_value(d, "driver_number"))
                if not num or num == "None":
                    continue
                drivers[num] = _best_driver_name(d)
                three_letter[num] = _pick_value(d, "three_letter_name")
                team_name = _pick_value(d, "team_name", "team")
                if team_name:
                    teams[num] = team_name

        except Exception:
            pass  # fall back if /drivers not available

        rows = []
        for r in qres:
            qpos = _pick_value(r, "position", "classification_position", "final_position")
            try:
                qpos = int(qpos) if qpos is not None and not (isinstance(qpos, float) and math.isnan(qpos)) else None
            except Exception:
                qpos = None
            if qpos is None:
                continue

            num  = str(_pick_value(r, "driver_number", "number", default=""))
            name = _best_driver_name(r) or drivers.get(num) or str(num)  # prefer session names; then /drivers; then number
            code = _pick_value(r, "three_letter_name") or three_letter.get(num) or _pick_value(r, "last_name") or num
            team = teams.get(num) or _pick_value(r, "team_name", "team")

            rows.append({"driver": name, "code": str(code), "grid_quali": qpos, "driver_number": num, "team": team})

        out = pd.DataFrame(rows).sort_values("grid_quali").reset_index(drop=True)

        # 4) Race starting grid -> grid (fallback to quali)
        out["grid"] = out["grid_quali"]
        if race is not None:
            race_key = _pick_value(race, "session_key", "key")
            try:
                g_resp = requests.get(f"{OPENF1_BASE}/starting_grid", params={"session_key": race_key}, timeout=25)
                g_resp.raise_for_status()
                gres = g_resp.json()
                by_num = {str(_pick_value(g, "driver_number")): _pick_value(g, "grid_position", "position") for g in gres if _pick_value(g, "driver_number") is not None}
                if by_num:
                    out["grid"] = out.apply(lambda r: by_num.get(str(r["driver_number"]), r["grid"]), axis=1)
            except Exception:
                pass

        out["grid"] = pd.to_numeric(out["grid"], errors="coerce").fillna(out["grid_quali"]).astype(int)
        return out[["driver","code","grid_quali","grid","team"]]

    except Exception as e:
        st.error(f"OpenF1 fetch failed: {e}")
        return pd.DataFrame(columns=["driver","code","grid_quali","grid"])

def fetch_latest_quali_openf1() -> pd.DataFrame:
    """
    Get the latest meeting’s Qualifying with proper driver names & codes.
    """
    try:
        s_resp = requests.get(f"{OPENF1_BASE}/sessions", params={"session_key": "latest"}, timeout=25)
        s_resp.raise_for_status()
        latest = s_resp.json()
        if isinstance(latest, list) and latest:
            latest = latest[0]
        meeting_key = _pick_value(latest, "meeting_key", "key", default=None)
        if not meeting_key:
            return pd.DataFrame(columns=["driver","code","grid_quali","grid"])

        s2 = requests.get(f"{OPENF1_BASE}/sessions", params={"meeting_key": meeting_key}, timeout=25).json()

        def _is_quali(s):
            name = str(_pick_value(s, "session_name", "name", default="")).lower()
            typ  = str(_pick_value(s, "session_type", "type", default="")).lower()
            return ("qualifying" in name) or ("qualifying" in typ) or ("sprint shootout" in name) or ("sprint_shootout" in typ)

        quali = next((s for s in s2 if _is_quali(s)), None)
        race  = next((s for s in s2 if str(_pick_value(s, "session_name", "name", default="")).lower() == "race"), None)
        if not quali:
            return pd.DataFrame(columns=["driver","code","grid_quali","grid"])

        quali_key = _pick_value(quali, "session_key", "key")

        # Quali classification
        qres = requests.get(f"{OPENF1_BASE}/session_result", params={"session_key": quali_key}, timeout=25).json()

        # Driver mapping for this session
        drivers = {}
        three_letter = {}
        teams = {}
        try:
            djson = requests.get(f"{OPENF1_BASE}/drivers", params={"session_key": quali_key}, timeout=25).json()
            for d in djson:
                num = str(_pick_value(d, "driver_number"))
                if not num or num == "None":
                    continue
                drivers[num] = _best_driver_name(d)
                three_letter[num] = _pick_value(d, "three_letter_name")
                team_name = _pick_value(d, "team_name", "team")
                if team_name:
                    teams[num] = team_name
        except Exception:
            pass

        rows = []
        for r in qres:
            pos = _pick_value(r, "position", "classification_position", "final_position")
            try:
                pos = int(pos) if pos is not None and not (isinstance(pos, float) and math.isnan(pos)) else None
            except Exception:
                pos = None
            if pos is None:
                continue

            num  = str(_pick_value(r, "driver_number", "number", default=""))
            name = _best_driver_name(r) or drivers.get(num) or str(num)
            code = _pick_value(r, "three_letter_name") or three_letter.get(num) or _pick_value(r, "last_name") or num
            team = teams.get(num) or _pick_value(r, "team_name", "team")
            rows.append({"driver": name, "code": str(code), "grid_quali": pos, "driver_number": num, "team": team})

        out = pd.DataFrame(rows).sort_values("grid_quali").reset_index(drop=True)
        out["grid"] = out["grid_quali"]

        if race is not None:
            try:
                gres = requests.get(f"{OPENF1_BASE}/starting_grid", params={"session_key": _pick_value(race, "session_key", "key")}, timeout=25).json()
                by_num = {str(_pick_value(g, "driver_number")): _pick_value(g, "grid_position", "position") for g in gres if _pick_value(g, "driver_number") is not None}
                if by_num:
                    out["grid"] = out.apply(lambda r: by_num.get(str(r["driver_number"]), r["grid"]), axis=1)
            except Exception:
                pass

        out["grid"] = pd.to_numeric(out["grid"], errors="coerce").fillna(out["grid_quali"]).astype(int)
        return out[["driver","code","grid_quali","grid","team"]]

    except Exception as e:
        st.error(f"OpenF1 latest fetch failed: {e}")
        return pd.DataFrame(columns=["driver","code","grid_quali","grid"])

SEASON_CIRCUIT_IDX = build_season_circuit_index()

def get_latest_meta_openf1():
    """
    Return {'season', 'round', 'label', 'country'} for the latest meeting.
    Prefers circuit short name (e.g., 'Zandvoort') for label.
    """
    try:
        # 1) Find the latest session → meeting_key
        s = requests.get(f"{OPENF1_BASE}/sessions", params={"session_key": "latest"}, timeout=20).json()
        if isinstance(s, list) and s:
            s = s[0]
        meeting_key = _pick_value(s, "meeting_key", "key")
        if not meeting_key:
            return {}

        # 2) Get the meeting record (best source of year/round/country/circuit)
        m = requests.get(f"{OPENF1_BASE}/meetings", params={"meeting_key": meeting_key}, timeout=20).json()
        if isinstance(m, list) and m:
            m = m[0]

        year    = _pick_value(m, "year", "season")
        rnd     = _pick_value(m, "round")
        country = _pick_value(m, "country", "country_name")
        # Prefer circuit short name (e.g. 'Zandvoort'). Fall back to meeting name.
        circuit = _pick_value(m, "circuit_short_name", "circuit_name", "circuit")
        mname   = _pick_value(m, "meeting_name", "name", "event_name")
        label   = circuit or mname or "Latest meeting"

        # 3) If round is missing, infer by ordering all meetings in that year
        if (rnd is None) and year:
            all_m = requests.get(f"{OPENF1_BASE}/meetings", params={"year": year}, timeout=20).json()
            def _date_key(mm):
                return _pick_value(mm, "date_start", "meeting_start_utc", "meeting_start_date", "date", default="")
            all_m_sorted = sorted(all_m, key=_date_key)
            for i, mm in enumerate(all_m_sorted, 1):
                if _pick_value(mm, "meeting_key", "key") == meeting_key:
                    rnd = i
                    break

        out = {
            "season": int(year) if year is not None else None,
            "round":  int(rnd)  if rnd  is not None else None,
            "label":  str(label),
            "country": country
        }
        return out
    except Exception:
        return {}



# Fantasy top 10
def _gumbel_sample_orders(scores: np.ndarray, n_sims: int, rng: np.random.Generator):
    n = scores.shape[0]
    g = rng.gumbel(loc=0.0, scale=1.0, size=(n_sims, n))  # Gumbel(0,1)
    noisy = g + scores[None, :]
    order = np.argsort(-noisy, axis=1)  # descending: winner first
    return order

def simulate_topk_from_winprob(win_prob: np.ndarray, n_sims: int = 5000, seed: int | None = None, k: int = 10):
    eps = 1e-6
    p = np.clip(win_prob.astype(float), eps, 1 - eps)
    scores = np.log(p)  # PL scores; only ratios matter
    rng = np.random.default_rng(seed)
    order = _gumbel_sample_orders(scores, n_sims, rng)

    n = p.shape[0]
    pos_of_driver = np.empty_like(order)
    row_idx = np.arange(n_sims)[:, None]
    pos_of_driver[row_idx, order] = np.arange(n)[None, :]
    exp_pos = pos_of_driver.mean(axis=0) + 1.0  # 1-based

    k = min(k, n)
    pos_counts = np.zeros((n, k), dtype=np.int32)
    for pos in range(k):
        idx_at_pos = order[:, pos]
        counts = np.bincount(idx_at_pos, minlength=n)
        pos_counts[:, pos] = counts

    pos_rate = (pos_counts / float(n_sims)) * 100.0
    return order, exp_pos, pos_counts, pos_rate

# F1 Fantasy Single driver contributions
def pretty_label(col: str) -> str:
    return NICE_LABELS.get(col, col)

def explain_logreg_percent(clf, feature_names, xvec):
    coefs = clf.coef_[0]
    raw = coefs * xvec
    denom = np.sum(np.abs(raw)) + 1e-9
    pct = (raw / denom) * 100.0
    dfc = pd.DataFrame({
        "Feature": [pretty_label(f) for f in feature_names],
        "Contribution %": pct
    }).sort_values("Contribution %", key=lambda s: s.abs(), ascending=False)
    return dfc

def parse_points_table(csv_text: str) -> Dict[int, int]:
    parts = [p.strip() for p in (csv_text or "").split(",") if p.strip()]
    return {i+1: int(v) for i, v in enumerate(parts)}

def positions_from_order(order: np.ndarray) -> np.ndarray:
    n_sims, n = order.shape
    pos = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    pos[rows, order] = np.arange(n)[None, :] + 1
    return pos

def expected_points_from_positions(pos_matrix: np.ndarray,
                                   pos_points: Dict[int, int],
                                   outside_points: int = 0) -> np.ndarray:
    n_sims, n = pos_matrix.shape
    max_p = pos_matrix.max()
    pts_vec = np.full((max_p + 1,), float(outside_points))
    for k, v in pos_points.items():
        if 1 <= k <= max_p:
            pts_vec[k] = float(v)
    pts = pts_vec[pos_matrix]
    return pts.mean(axis=0)

# === Model quality / backtest helpers ===
def _fit_fold_lr(train_df: pd.DataFrame, feature_cols: list[str], calibrate: bool):
    """Fit a Logistic Regression on the training fold (optionally isotonic-calibrated)."""
    # Per-fold defaults (avoid data leakage)
    fold_defaults = train_df[feature_cols].mean(numeric_only=True).fillna(0.0)

    Xtr_df = train_df[feature_cols].astype(float).fillna(fold_defaults).fillna(0.0)
    # Final sanitation: kill any lingering NaN/Inf
    Xtr = np.nan_to_num(Xtr_df.values, nan=0.0, posinf=0.0, neginf=0.0)

    ytr = pd.to_numeric(train_df["win"], errors="coerce").fillna(0).astype(int).values

    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5) if calibrate else base
    clf.fit(Xtr, ytr)
    return clf, fold_defaults



def _predict_fold(
    clf,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    fold_defaults: pd.Series,
    *,
    method: str = "lr",                  # "lr" | "softmax" | "blend"
    grid_alpha: float = 0.08,
    gap_beta: float = 0.06,
    temp_T: float = 1.25,
    blend: float = 0.6,
) -> np.ndarray:
    """Predict probabilities for a test fold. If method != 'lr', normalizes per-race."""
    # Work on a positional index so grouping returns integer positions
    tdf = test_df.reset_index(drop=True)

    # Build X (NaN/Inf safe)
    Xte_df = tdf[feature_cols].astype(float).fillna(fold_defaults).fillna(0.0)
    Xte = np.nan_to_num(Xte_df.values, nan=0.0, posinf=0.0, neginf=0.0)

    # Plain LR (fast path)
    if method == "lr" or "raceId" not in tdf.columns:
        return clf.predict_proba(Xte)[:, 1]

    # Precompute pieces used by softmax/blend
    # Base logits: decision_function if available, else logit(p)
    if hasattr(clf, "decision_function"):
        logits_all = clf.decision_function(Xte)
        logits_all = np.asarray(logits_all).reshape(-1)  # binary -> (n,)
    else:
        p_all = clf.predict_proba(Xte)[:, 1]
        eps = 1e-6
        p_clip = np.clip(p_all, eps, 1 - eps)
        logits_all = np.log(p_clip / (1 - p_clip))
    # For blend we also need LR probs
    p_lr_all = clf.predict_proba(Xte)[:, 1] if method == "blend" else None

    # Group by race and normalize within each race
    out = np.zeros(len(tdf), dtype=float)
    groups = tdf.groupby("raceId", sort=False).indices  # dict: raceId -> array of positions
    for _, pos in groups.items():
        pos = np.asarray(pos, dtype=int)
        scores = logits_all[pos].copy()

        # optional tilts
        if "grid" in tdf.columns:
            g = pd.to_numeric(tdf.loc[pos, "grid"], errors="coerce").to_numpy(dtype=float)
            g = np.nan_to_num(g, nan=np.nanmean(g) if np.isfinite(g).any() else 1.0)
            scores += (-grid_alpha) * (g - 1.0)
        if {"grid", "grid_quali"}.issubset(tdf.columns):
            sp = (
                pd.to_numeric(tdf.loc[pos, "grid"], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(tdf.loc[pos, "grid_quali"], errors="coerce").to_numpy(dtype=float)
            )
            sp = np.nan_to_num(sp, nan=0.0)
            scores += (-gap_beta) * sp

        # normalize to per-race win probabilities
        scores = scores - np.max(scores)  # stability
        expu = np.exp(scores / max(temp_T, 1e-6))
        p_sm = expu / max(expu.sum(), 1e-9)

        if method == "softmax":
            p_r = p_sm
        elif method == "blend":
            p_lr = p_lr_all[pos]
            p_lr_norm = p_lr / max(p_lr.sum(), 1e-9)
            p_r = (1.0 - blend) * p_lr_norm + blend * p_sm
        else:
            p_r = clf.predict_proba(Xte[pos, :])[:, 1]

        out[pos] = p_r

    return out

def recommend_team(
    fantasy_df: pd.DataFrame,
    constructor_df: pd.DataFrame | None = None,
    *,
    n_drivers: int = 5,
    n_cons: int = 1,
    budget: float | None = None,
    price_col_driver: str = "price",
    price_col_cons: str = "price",
    drs_mult: float = 2.0,
    topN: int = 15,
):
    """
    Greedy team builder that:
      - sorts drivers by exp_points then win_prob (win_pct fallback)
      - optionally enforces a budget
      - picks constructors greedily after drivers
      - suggests a DRS target = highest exp_points among chosen drivers
    It never crashes if win_prob/price/team are missing.
    """
    if fantasy_df is None or fantasy_df.empty:
        return None, []

    df = fantasy_df.copy()

    # --- Normalize required columns ---
    if "driver" not in df.columns:
        df["driver"] = df.index.astype(str)

    if "exp_points" not in df.columns:
        raise ValueError("fantasy_df must include 'exp_points' (expected points per driver).")

    if "win_prob" not in df.columns:
        if "win_pct" in df.columns:
            df["win_prob"] = pd.to_numeric(df["win_pct"], errors="coerce") / 100.0
        else:
            df["win_prob"] = 0.0

    if price_col_driver not in df.columns:
        df[price_col_driver] = 0.0

    if "team" not in df.columns:
        df["team"] = ""

    # keep only topN by (exp_points, win_prob)
    df = (df
          .sort_values(["exp_points", "win_prob"], ascending=[False, False])
          .head(int(topN))
          .reset_index(drop=True))

    # --- Pick drivers greedily under budget ---
    chosen_drivers = []
    cost_drivers = 0.0
    pts_total = 0.0

    # minimal constructor budget reservation (so we don't overspend on drivers)
    min_cons_cost = 0.0
    cons_pool = None
    if (constructor_df is not None) and (n_cons > 0):
        cons_pool = constructor_df.copy()
        if cons_pool.empty:
            cons_pool = None
        else:
            if price_col_cons not in cons_pool.columns:
                cons_pool[price_col_cons] = 0.0
            if "exp_points" not in cons_pool.columns:
                cons_pool["exp_points"] = 0.0
            # reserve cheapest n_cons cost if a budget exists
            if budget is not None:
                min_cons_cost = float(cons_pool[price_col_cons].nsmallest(n_cons).sum())

    for _, r in df.iterrows():
        if len(chosen_drivers) >= n_drivers:
            break
        proposed = cost_drivers + float(r[price_col_driver])
        if budget is not None:
            if proposed + min_cons_cost > float(budget) + 1e-9:
                continue
        chosen_drivers.append(r.to_dict())
        cost_drivers = proposed
        pts_total += float(r["exp_points"])

    # If we couldn't fill driver slots under the cap, bail out
    if len(chosen_drivers) < n_drivers:
        return None, []

    # --- Pick constructors greedily with remaining budget ---
    chosen_cons = []
    cost_cons = 0.0
    if (cons_pool is not None) and (n_cons > 0):
        cons_pool = cons_pool.sort_values("exp_points", ascending=False).reset_index(drop=True)
        rem = None if budget is None else (float(budget) - cost_drivers)
        for _, r in cons_pool.iterrows():
            if len(chosen_cons) >= n_cons:
                break
            price = float(r[price_col_cons])
            if (rem is None) or (price <= rem + 1e-9):
                chosen_cons.append(r.to_dict())
                cost_cons += price
                pts_total += float(r["exp_points"])
                if rem is not None:
                    rem -= price

        if len(chosen_cons) < n_cons:
            # couldn't afford enough constructors
            return None, []

    # --- DRS recommendation within chosen drivers ---
    drs_pick = None
    if drs_mult and drs_mult > 1.0 and chosen_drivers:
        drs_row = max(chosen_drivers, key=lambda d: float(d.get("exp_points", 0.0)))
        drs_pick = drs_row.get("driver")
        pts_total += (float(drs_mult) - 1.0) * float(drs_row.get("exp_points", 0.0))

    best = {
        "drivers": [
            {
                "driver": d.get("driver"),
                "team": d.get("team", ""),
                "exp_points": float(d.get("exp_points", 0.0)),
                "win_prob": float(d.get("win_prob", 0.0)),
                "price": float(d.get(price_col_driver, 0.0)),
            } for d in chosen_drivers
        ],
        "constructors": [
            {
                "team": c.get("team"),
                "exp_points": float(c.get("exp_points", 0.0)),
                "price": float(c.get(price_col_cons, 0.0)),
            } for c in chosen_cons
        ],
        "points": float(pts_total),
        "cost": float(cost_drivers + cost_cons),
        "drs_pick": drs_pick,
    }

    # (Optional) Return alternatives list; keeping empty for now
    return best, []

def build_latest_constructor_map(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    For each driverId, take the most recent row in your processed DF and return:
    driverId, constructorId, team (readable).
    Works even if your processed DF uses constructorRef/constructor/name.
    """
    keep = [c for c in ["driverId","constructorId","constructorRef","constructor","team","season","round"] if c in df_all.columns]
    if "driverId" not in keep:
        return pd.DataFrame(columns=["driverId","constructorId","team"])

    d = df_all[keep].copy()
    # latest row per driver
    if {"season","round"}.issubset(d.columns):
        d = d.sort_values(["driverId","season","round"])
    else:
        d = d.sort_values(["driverId"])

    latest = d.groupby("driverId", as_index=False).last()

    # Normalize team name
    if "team" not in latest.columns:
        if "constructor" in latest.columns:
            latest = latest.rename(columns={"constructor": "team"})
        elif "constructorRef" in latest.columns:
            latest["team"] = latest["constructorRef"].astype(str).str.replace("-", " ").str.title()
        else:
            latest["team"] = ""

    # Try to improve team name from data/raw/constructors.csv
    cp = Path("data/raw/constructors.csv")
    if cp.exists():
        con = pd.read_csv(cp)
        lower = {c.lower(): c for c in con.columns}
        cid = lower.get("constructorid", "constructorId")
        cname = lower.get("name", "name")
        if cid in con.columns and cname in con.columns and "constructorId" in latest.columns:
            latest = latest.merge(
                con[[cid, cname]].rename(columns={cid: "constructorId", cname: "team_name"}),
                on="constructorId", how="left"
            )
            latest["team"] = latest["team"].fillna(latest["team_name"])
            latest = latest.drop(columns=[c for c in ["team_name"] if c in latest.columns])

    latest["team"] = latest["team"].fillna("").astype(str)
    if "constructorId" not in latest.columns:
        latest["constructorId"] = pd.NA
    return latest[["driverId","constructorId","team"]].drop_duplicates("driverId")




def _top1_hit_rate(test_df: pd.DataFrame, p: np.ndarray) -> tuple[float, int, int]:
    """Top-1 hit rate at the race level (did our highest-prob driver win?)."""
    tmp = test_df[["raceId", "win"]].copy()
    tmp["p"] = p
    hits = 0
    races = 0
    for rid, g in tmp.groupby("raceId"):
        if g.empty:
            continue
        idx = g["p"].idxmax()
        hits += int(g.loc[idx, "win"] == 1)
        races += 1
    rate = hits / races if races else 0.0
    return rate, hits, races


def backtest_by_season(df_all: pd.DataFrame, feature_cols: list[str], calibrate: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Rolling-season backtest: for each season S, train on seasons < S and test on S.
    Returns (per_season_df, overall_summary_dict).
    """
    df_all = df_all.loc[:, ~df_all.columns.duplicated()].copy()

    needed = {"season", "raceId", "win", *feature_cols}
    if not needed.issubset(df_all.columns):
        missing = sorted(list(needed - set(df_all.columns)))
        raise ValueError(f"Backtest needs columns missing in df: {missing}")

    df_use = df_all.dropna(subset=["season", "raceId"]).copy()
    seasons = sorted(pd.to_numeric(df_use["season"], errors="coerce").dropna().unique().astype(int))

    rows = []
    all_hits = all_races = 0
    w_logloss = w_brier = 0.0
    n_rows_total = 0

    for s in seasons:
        train = df_use[df_use["season"] < s].copy()
        test  = df_use[df_use["season"] == s].copy()

        # Skip very early seasons with no prior training data
        if train["raceId"].nunique() < 2 or len(train) < 100 or test.empty:
            continue

        clf_fold, fold_defaults = _fit_fold_lr(train, feature_cols, calibrate=calibrate)
        p = _predict_fold(
        clf_fold, test, feature_cols, fold_defaults,
        method="blend", grid_alpha=0.08, gap_beta=0.06, temp_T=1.25, blend=0.6,
        )


        y = test["win"].astype(int).values
        try:
            ll = log_loss(y, p, labels=[0,1])
        except ValueError:
            # Edge case: only one class present
            ll = np.nan
        br = brier_score_loss(y, p)

        top1, hits, races = _top1_hit_rate(test, p)

        rows.append({
            "season": s,
            "n_rows": len(test),
            "n_races": races,
            "log_loss": ll,
            "brier": br,
            "top1_hit": top1,
            "avg_pred_win_pct": float(np.mean(p))*100.0,
        })

        # accumulate weighted
        n_rows_total += len(test)
        if not np.isnan(ll):
            w_logloss += ll * len(test)
        w_brier  += br * len(test)
        all_hits += hits
        all_races += races

    per_season = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)

    overall = {
        "seasons_evaluated": int(per_season["season"].nunique()) if not per_season.empty else 0,
        "n_rows_total": int(n_rows_total),
        "n_races_total": int(all_races),
        "log_loss_weighted": (w_logloss / n_rows_total) if n_rows_total else np.nan,
        "brier_weighted":    (w_brier  / n_rows_total) if n_rows_total else np.nan,
        "top1_hit_overall":  (all_hits / all_races) if all_races else np.nan,
    }
    return per_season, overall

def _softmax_vec(scores: np.ndarray, T: float = 1.0) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s / max(T, 1e-6)
    s = s - np.max(s)  # for numerical stability
    e = np.exp(s)
    denom = e.sum()
    return e / (denom if denom > 0 else 1.0)

def _safe_logits(clf, X: np.ndarray) -> np.ndarray:
    """Return a real-valued score for each row, even for calibrated models."""
    # Preferred: raw margins if available
    if hasattr(clf, "decision_function"):
        z = clf.decision_function(X)
        # Some classifiers return shape (n_samples, n_classes); use the positive class
        if isinstance(z, np.ndarray) and z.ndim == 2 and z.shape[1] == 2:
            z = z[:, 1]
        return z.astype(float)

    # Fallback: logit of predicted probability
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)[:, 1]
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        return np.log(p) - np.log(1 - p)

    # Last resort: decision_function/predict_proba not present — use zeros
    return np.zeros(X.shape[0], dtype=float)


def compute_win_probs(
    X: np.ndarray,
    *,
    clf,
    grid: Iterable | None = None,
    start_penalty: Iterable | None = None,  # grid - quali
    method: str = "blend",                  # "lr" | "softmax" | "blend"
    grid_alpha: float = 0.08,
    gap_beta: float = 0.06,
    temp_T: float = 1.25,
    blend: float = 0.6
) -> np.ndarray:
    base_logits = _safe_logits(clf, X).astype(float)
    scores = np.nan_to_num(base_logits, nan=0.0, posinf=0.0, neginf=0.0)

    if method in ("softmax", "blend"):
        if grid is not None:
            g = np.asarray(grid, dtype=float)
            if np.isnan(g).any():
                # fallback: replace NaNs with race-local mean or 1.0
                mean_g = np.nanmean(g) if np.isfinite(np.nanmean(g)) else 1.0
                g = np.where(np.isnan(g), mean_g, g)
            scores += (-grid_alpha) * (g - 1.0)
        if start_penalty is not None:
            sp = np.asarray(start_penalty, dtype=float)
            sp = np.nan_to_num(sp, nan=0.0, posinf=0.0, neginf=0.0)
            scores += (-gap_beta) * sp

    if method == "softmax":
        p = _softmax_vec(scores, T=temp_T)
    elif method == "blend":
        p_lr = clf.predict_proba(X)[:, 1]
        p_lr = np.nan_to_num(p_lr, nan=0.0)
        denom = max(p_lr.sum(), 1e-9)
        p_lr_norm = p_lr / denom
        p_sm = _softmax_vec(scores, T=temp_T)
        p = (1.0 - blend) * p_lr_norm + blend * p_sm
    else:  # "lr"
        p = clf.predict_proba(X)[:, 1]

    return np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)


def _race_winner_neglogloss(test_df: pd.DataFrame, p: np.ndarray) -> float:
    """Average -log(p_winner) per race; lower is better."""
    tmp = test_df[["raceId", "win"]].copy()
    tmp["p"] = p
    losses = []
    for rid, g in tmp.groupby("raceId"):
        if g.empty: 
            continue
        # probability assigned to the actual winner in this race
        pw = g.loc[g["win"] == 1, "p"]
        if pw.empty:
            continue
        prob = float(pw.iloc[0])
        losses.append(-np.log(max(prob, 1e-15)))
    return float(np.mean(losses)) if losses else np.nan

def backtest_by_season_with_combiner(
    df_all: pd.DataFrame, feature_cols: list[str],
    *, temp_T=1.25, blend=0.6, grid_alpha=0.08, gap_beta=0.06
) -> tuple[pd.DataFrame, dict]:
    needed = {"season","raceId","win", *feature_cols}
    if not needed.issubset(df_all.columns):
        missing = sorted(list(needed - set(df_all.columns)))
        raise ValueError(f"Backtest needs columns missing in df: {missing}")

    df_use = df_all.dropna(subset=["season","raceId"]).copy()
    seasons = sorted(pd.to_numeric(df_use["season"], errors="coerce").dropna().unique().astype(int))

    rows = []
    all_hits = all_races = 0
    w_brier = 0.0
    n_rows_total = 0
    nll_sum = 0.0
    n_races_for_nll = 0

    for s in seasons:
        train = df_use[df_use["season"] < s].copy()
        test  = df_use[df_use["season"] == s].copy()
        if train["raceId"].nunique() < 2 or len(train) < 100 or test.empty:
            continue

        clf_fold, fold_defaults = _fit_fold_lr(train, feature_cols, calibrate=False)

        # ✅ Work on a reset-index view so group indices are POSITIONS (0..n-1)
        t = test.reset_index(drop=True).copy()

        # Build X for this fold (NaN/Inf safe)
        Xte_df = t.reindex(columns=feature_cols).astype(float).fillna(fold_defaults).fillna(0.0)
        Xte = np.nan_to_num(Xte_df.values, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict per race using positional indices
        p = np.zeros(len(t), dtype=float)
        groups = t.groupby("raceId", sort=False).indices  # {raceId: array of POSITIONS}
        for _, idx in groups.items():
            idx = np.asarray(idx, dtype=int)  # positions into t / Xte
            X_r = Xte[idx, :]

            g = None
            sp = None
            if "grid" in t.columns:
                g_arr = pd.to_numeric(t.loc[idx, "grid"], errors="coerce").to_numpy(dtype=float)
                mg = np.nanmean(g_arr) if np.isfinite(np.nanmean(g_arr)) else 1.0
                g = np.where(np.isnan(g_arr), mg, g_arr)

            if {"grid","grid_quali"}.issubset(t.columns):
                q_arr = pd.to_numeric(t.loc[idx, "grid_quali"], errors="coerce").to_numpy(dtype=float)
                sp = np.nan_to_num((g if g is not None else pd.to_numeric(t.loc[idx, "grid"], errors="coerce").to_numpy(dtype=float)) - q_arr,
                                   nan=0.0, posinf=0.0, neginf=0.0)

            p[idx] = compute_win_probs(
                X_r, clf=clf_fold,
                grid=g, start_penalty=sp,
                method="blend", grid_alpha=grid_alpha, gap_beta=gap_beta, temp_T=temp_T, blend=blend
            )

        # Metrics on the reset-index frame `t`
        y = pd.to_numeric(t["win"], errors="coerce").fillna(0).astype(int).to_numpy()
        br = brier_score_loss(y, np.nan_to_num(p, nan=0.0))
        top1, hits, races = _top1_hit_rate(t, p)
        nll = _race_winner_neglogloss(t, p)

        rows.append({
            "season": s, "n_rows": len(t), "n_races": races,
            "brier": br, "top1_hit": top1, "race_neglogloss": nll
        })

        n_rows_total += len(t)
        w_brier += br * len(t)
        all_hits += hits
        all_races += races
        if not np.isnan(nll):
            nll_sum += nll * max(races, 1)
            n_races_for_nll += max(races, 1)

    per_season = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
    overall = {
        "seasons_evaluated": int(per_season["season"].nunique()) if not per_season.empty else 0,
        "n_rows_total": int(n_rows_total),
        "n_races_total": int(all_races),
        "brier_weighted": (w_brier / n_rows_total) if n_rows_total else np.nan,
        "top1_hit_overall": (all_hits / all_races) if all_races else np.nan,
        "race_neglogloss_overall": (nll_sum / n_races_for_nll) if n_races_for_nll else np.nan,
        "combiner": dict(temp_T=temp_T, blend=blend, grid_alpha=grid_alpha, gap_beta=gap_beta),
    }
    return per_season, overall


def autotune_combiner(df_all, feature_cols):
    df_all = df_all.loc[:, ~df_all.columns.duplicated()].copy()
    feature_cols = list(pd.Index(feature_cols).drop_duplicates())
    grid_T      = [0.9, 1.0, 1.15, 1.25, 1.4]
    grid_blend  = [0.3, 0.5, 0.6, 0.7]
    grid_galpha = [0.04, 0.06, 0.08, 0.10]
    grid_gbeta  = [0.00, 0.03, 0.06, 0.09]

    best = None
    best_score = float("inf")   # lower = better (race_neglogloss)
    for T in grid_T:
        for blend in grid_blend:
            for ga in grid_galpha:
                for gb in grid_gbeta:
                    _, overall = backtest_by_season_with_combiner(
                        df_all, feature_cols, temp_T=T, blend=blend, grid_alpha=ga, gap_beta=gb
                    )
                    nll = overall.get("race_neglogloss_overall", np.nan)
                    if np.isnan(nll): 
                        continue
                    if nll < best_score:
                        best_score = nll
                        best = overall["combiner"]
                        best["score_nll"] = nll
                        best["top1"] = overall.get("top1_hit_overall", np.nan)
                        best["brier"] = overall.get("brier_weighted", np.nan)
    return best

def _fold_ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower().strip()

BY_NAME_FOLDED = { _fold_ascii(k): v for k, v in (BY_NAME or {}).items() }

def map_row_to_driver_id(row) -> int | None:
    # 1) code (most reliable)
    c = str(row.get("code") or "").upper()
    if BY_CODE and c in BY_CODE:
        return BY_CODE[c]
    # 2) full name (accent/space-insensitive)
    n = row.get("driver")
    if n and BY_NAME_FOLDED:
        return BY_NAME_FOLDED.get(_fold_ascii(n))
    return None


def _force_int64_key(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


def _strip_accents(s: str) -> str:
    if not isinstance(s, str): return ""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def build_surname_map() -> dict[str, int]:
    """Surname -> driverId (only if surname is unique in drivers.csv)."""
    drv_p = Path("data/raw/drivers.csv")
    if not drv_p.exists(): return {}
    drv = pd.read_csv(drv_p)

    lower = {c.lower(): c for c in drv.columns}
    idc = lower.get("driverid")
    snc = lower.get("surname") or lower.get("familyname")
    if idc is None or snc is None: return {}

    tmp = pd.DataFrame({
        "id": pd.to_numeric(drv[idc], errors="coerce"),
        "surname": drv[snc].astype(str).map(_strip_accents).str.lower().str.strip()
    }).dropna()
    vc = tmp["surname"].value_counts()
    uniq = set(vc[vc == 1].index)  # only keep surnames that appear once historically
    return {s: int(i) for s, i in zip(tmp["surname"], tmp["id"]) if s in uniq}

SURNAME_TO_ID = build_surname_map()

def map_driver_to_id_from_row(name: str | None, code: str | None = None):
    """
    Robust mapper: try 3-letter code, full name, then 'initial SURNAMe' → surname.
    """
    # 1) 3-letter code (OpenF1 three_letter_name)
    if isinstance(code, str) and BY_CODE:
        c = code.strip().upper()
        if c in BY_CODE:
            return BY_CODE[c]

    # 2) Full name exact match (from drivers.csv)
    if isinstance(name, str) and BY_NAME:
        n = _strip_accents(name).strip().lower()
        if n in BY_NAME:
            return BY_NAME[n]

        # 3) Handle formats like "O PIASTRI" → surname only
        tokens = re.sub(r"[^A-Za-z]+", " ", n).split()
        if tokens:
            sur = tokens[-1]  # last token is the surname
            if SURNAME_TO_ID and sur in SURNAME_TO_ID:
                return SURNAME_TO_ID[sur]

    return None

# CSS
st.markdown("""
<style>
/* Banner with title on its own line, chips always on next line */
.banner {
  display: flex;
  flex-wrap: wrap;           /* allow wrapping */
  align-items: center;
  gap: .6rem;
  margin-bottom: 6px;
}

/* Title always occupies the full first line */
.banner h3 {
  margin: 0;
  flex-basis: 100%;          /* forces the title to be a full-width row */
  white-space: nowrap;
}

/* Chips row always starts on a new line */
.banner .chips {
  display: flex;
  flex-wrap: wrap;
  gap: .6rem;
  width: 100%;               /* take full width -> guarantees a new line */
  margin-top: 2px;
}

/* Round and weather chips */
.round-chip, .wx-pill {
  background: #1f2633;
  border-radius: 999px;
  padding: 4px 10px;
  font-weight: 700;
  white-space: nowrap;
}

.wx-pill {
  background: #2a3242;
  font-size: 1rem;
}

details > div, .streamlit-expanderContent {
  padding-top: 4px !important;
}

/* ---- Sidebar typography ---- */
:is(aside, section)[data-testid="stSidebar"] {}  /* keep parser happy */

/* Sidebar headers (st.sidebar.header/subheader) */
:is(aside, section)[data-testid="stSidebar"] h2,
:is(aside, section)[data-testid="stSidebar"] h3 {
  font-size: 2.35rem !important;
  font-weight: 800 !important;
  letter-spacing: .2px;
}

/* Widget labels (Selectbox/Radio/Toggle/Button labels, etc.) */
:is(aside, section)[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label p {
  font-size: 1.50rem !important;
  font-weight: 700 !important;
  line-height: 1.35 !important;
  margin-bottom: .15rem !important;
}

/* Radio options / toggle text */
:is(aside, section)[data-testid="stSidebar"] [role="radiogroup"] label p {
  font-size: 1.36rem !important;
}

/* Selectbox chosen value & dropdown input text */
:is(aside, section)[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  font-size: 1.25rem !important;
}

/* Inputs (text, number, textarea) */
:is(aside, section)[data-testid="stSidebar"] input,
:is(aside, section)[data-testid="stSidebar"] textarea {
  font-size: 1.56rem !important;
}

/* Buttons in sidebar */
:is(aside, section)[data-testid="stSidebar"] button {
  font-size: 1.56rem !important;
  padding: 0.5rem 0.75rem !important;
}

/* Slider ticks & value bubble */
:is(aside, section)[data-testid="stSidebar"] [data-testid="stTickBarMin"],
:is(aside, section)[data-testid="stSidebar"] [data-testid="stTickBarMax"],
:is(aside, section)[data-testid="stSidebar"] [data-testid="stSliderValue"] {
  font-size: 1.00rem !important;
}
/* --- Make sidebar toggles & radio bullets bigger --- */

/* Ensure scaled controls aren't clipped */
:is(aside, section)[data-testid="stSidebar"] [data-testid="stWidget"] { 
  overflow: visible; 
}

/* Enlarge the TOGGLE switch (st.toggle) */
:is(aside, section)[data-testid="stSidebar"] [data-testid="stToggle"] label > div:first-child {
  transform: scale(1.25);
  transform-origin: left center;
}

/* Fallback for some Streamlit/BaseWeb builds */
:is(aside, section)[data-testid="stSidebar"] [data-baseweb="switch"] {
  transform: scale(1.25);
  transform-origin: left center;
}

/* Enlarge RADIO bullets (st.radio) */
:is(aside, section)[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
  transform: scale(1.2);
  transform-origin: left center;
  margin-right: .4rem; /* keep spacing nice */
}

/* Slightly increase row height for easier tapping */
:is(aside, section)[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: .15rem 0;
}

/* --- Align + size sidebar toggles & radio options --- */

/* Keep scaled controls from being clipped */
:is(aside, section)[data-testid="stSidebar"] [data-testid="stWidget"] { 
  overflow: visible; 
}

/* TOGGLES (st.toggle) — put control + text on one row and center vertically */
:is(aside, section)[data-testid="stSidebar"] [data-testid="stToggle"] label {
  display: flex !important;
  align-items: center !important;
  gap: .45rem;
}

/* Enlarge the switch and nudge it so it's visually centered with the text */
:is(aside, section)[data-testid="stSidebar"] [data-testid="stToggle"] label > div:first-child {
  transform: scale(1.25) translateY(1px); /* tweak translateY to 0/2px if needed */
  transform-origin: left center;
}

/* RADIO (st.radio) — center bullets with text */
:is(aside, section)[data-testid="stSidebar"] [role="radiogroup"] label {
  display: flex !important;
  align-items: center !important;
  gap: .45rem;
}

/* Enlarge the bullet and nudge it to align */
:is(aside, section)[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
  transform: scale(1.2) translateY(1px);  /* adjust to taste */
  transform-origin: left center;
}

/* Tighten label text vertical metrics so they center nicely */
:is(aside, section)[data-testid="stSidebar"] [role="radiogroup"] label p,
:is(aside, section)[data-testid="stSidebar"] [data-testid="stToggle"] label p {
  margin: 0 !important;
  line-height: 1.2 !important;
}

/* ===== Main content typography (not sidebar) ===== */
section[data-testid="stMain"] h1 {
  font-size: 2.0rem !important;
  font-weight: 800;
  margin: 0 0 .6rem;
}
section[data-testid="stMain"] h2 {
  font-size: 1.45rem !important;
  font-weight: 800;
  margin: 1.0rem 0 .4rem;
}
section[data-testid="stMain"] h3 {
  font-size: 1.15rem !important;
  font-weight: 700;
  margin: .75rem 0 .3rem;
}

/* Your race banner title should sit between h2 and h3 */
.banner h3 {
  font-size: 1.35rem !important;
  font-weight: 800;
}

/* Make the "Tweak inputs" custom title slightly smaller than h2 */
section[data-testid="stMain"] .whatif-title {
  font-size: 1.30rem !important;
  font-weight: 700;
  letter-spacing: .2px;
}

/* Dataframe readability: header/body font sizes & spacing */
section[data-testid="stMain"] [data-testid="stDataFrame"] {
  margin: .25rem 0 1.1rem;
}
section[data-testid="stMain"] [data-testid="stDataFrame"] div[role="columnheader"] {
  font-size: .96rem !important;
  font-weight: 700 !important;
}
section[data-testid="stMain"] [data-testid="stDataFrame"] div[role="gridcell"] {
  font-size: .95rem !important;
}

/* Captions a touch smaller and tighter */
section[data-testid="stMain"] [data-testid="stCaptionContainer"] p {
  font-size: .92rem !important;
  line-height: 1.25 !important;
  margin-top: .25rem !important;
}

</style>
""", unsafe_allow_html=True)




st.title("🏁 F1 Next-Race Winner Predictor")
st.info(
    "**Quick start**\n"
    "1) Pick **Season** and **Circuit** in the left sidebar.\n"
    "2) Choose a **Grid data source** (try *Fetch latest qualifying*).\n"
    "3) Open **Predictions** tab to see win %.\n"
    "4) Use **Tweak inputs** to explore wet/hot and grid changes.",
    icon="👉"
)




DATA_PATH = Path("data/processed/baseline_with_features.csv")
if not DATA_PATH.exists():
    st.error("Missing data/processed/baseline_with_features.csv. Run the notebook section that saves df_feat.")
    st.stop()

df = load_csv(DATA_PATH)
df = df.loc[:, ~df.columns.duplicated()].copy()

# ---- First-load default: use latest qualifying + meta (so the banner knows) ----
if "grid_src" not in st.session_state:
    try:
        dfq = fetch_latest_quali_openf1()
        if not dfq.empty:
            st.session_state["grid_df"] = dfq
            st.session_state["grid_src"] = "openf1_latest"
            # cache latest meeting info for the banner
            if "latest_meta" not in st.session_state:
                st.session_state["latest_meta"] = get_latest_meta_openf1()
    except Exception:
        # If OpenF1 is down, do nothing and fall back to sidebar season/round
        pass



# --- inject season/round from data/raw/races.csv if missing ---
raw = Path("data/raw")
races_p = raw / "races.csv"

if races_p.exists():
    races_raw = pd.read_csv(races_p)
    # normalize common cols
    lc = {c.lower(): c for c in races_raw.columns}
    def col(*names):
        for n in names:
            if n in lc: return lc[n]
        return None

    rnorm = races_raw.rename(columns={
        col("raceid"): "raceId",
        col("year"):   "season",
    })
    rc = col("round")
    if rc and rc != "round":
        rnorm = rnorm.rename(columns={rc: "round"})

    # Keep only columns that df is missing (avoid suffix collisions)
    candidate_cols = ["season", "round", "circuitId", "name"]
    to_add = [c for c in candidate_cols if (c in rnorm.columns) and (c not in df.columns)]
    if to_add:
        slim = rnorm[["raceId"] + to_add].drop_duplicates("raceId")
        df = df.merge(slim, on="raceId", how="left")



# --- weather helpers & quick debug ---
wx_cols = [
    "wx_temp_mean_c","wx_temp_max_c","wx_precip_mm","wx_rain_mm",
    "wx_windspeed_max_kmh","wx_windgusts_max_kmh",
    "wx_rh_mean_pct","wx_cloud_mean_pct","wx_is_wet","wx_is_hot"
]
has_wx = any(c in df.columns for c in wx_cols)

show_dbg = st.sidebar.toggle("Show debug info", value=False)
if show_dbg:
    with st.expander("Debug info"):
        st.caption(f"Loaded: {DATA_PATH.resolve()}")
        st.caption(f"wx cols found: {sorted([c for c in df.columns if c.startswith('wx_')])}")
        st.caption(f"has season/round? { {'season','round'}.issubset(df.columns) }")
        if has_wx:
            cov = df[[c for c in wx_cols if c in df.columns]].notna().mean().round(3)
            st.caption(f"wx coverage (fraction non-null): {cov.to_dict()}")


df["position"] = pd.to_numeric(df.get("position"), errors="coerce")

# Coerce any feature columns we might use
num_cols = [
    "grid","grid_quali",
    "drv_pts_3","team_pts_3","drv_pos_3","drv_pts_ytd","team_pts_ytd",
    "track_podium_rate","track_dnf_rate","qual_gap_to_pole",
    "grid_sq","grid_log","front_row",
    "grid_quali_sq","grid_quali_log","start_penalty",
    "drv_starts_cum","drv_wins_cum","drv_podiums_cum","drv_points_cum",
    "drv_win_rate_career","drv_podium_rate_career","drv_pts_per_start_career",
    "drv_years_exp_so_far",'drv_track_starts_cum', 'drv_track_podiums_cum', 'drv_track_wins_cum',
    'drv_track_podium_rate', 'drv_track_win_rate', "wx_temp_mean_c","wx_temp_max_c", "wx_precip_mm","wx_rain_mm",
    "wx_windspeed_max_kmh","wx_windgusts_max_kmh",
    "wx_rh_mean_pct","wx_cloud_mean_pct",
    "wx_is_wet","wx_is_hot"
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")


# keep finished rows only
df = df[df["position"].notna()].copy()

# target
df["win"] = (df["position"] == 1).astype(int)

# Use Tier-1 features if present (fallback to grid)
feature_cols = list(pd.Index([c for c in num_cols if c in df.columns]).drop_duplicates())
if not feature_cols:
    feature_cols = ["grid"]  # last resort

# Build defaults from training data (means), then ensure no NaNs remain
feature_defaults = df[feature_cols].mean(numeric_only=True).fillna(0.0)

X = df[feature_cols].astype(float).fillna(feature_defaults).fillna(0.0)
y = df["win"].astype(int)

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf = train_lr(X, y)

# --- Build latest driver-specific feature reference from the enriched CSV ---
driver_feature_cols = [c for c in feature_cols if c.startswith("drv_")]  # e.g., drv_win_rate_career, etc.
driver_feats_ref = pd.DataFrame()

if driver_feature_cols and "driverId" in df.columns:
    # Sort so we can take the most recent row per driver
    if {"season","round"}.issubset(df.columns):
        df_sorted = df.sort_values(["driverId","season","round"])
    else:
        df_sorted = df.sort_values(["driverId"])
    driver_feats_ref = (
        df_sorted.groupby("driverId", as_index=False)[driver_feature_cols].last()
        .copy()
    )

# --- Latest driver×circuit (track-affinity) features ---
driver_circuit_feature_cols = [c for c in feature_cols if c.startswith("drv_track_")]
driver_circuit_feats_ref = pd.DataFrame()

if driver_circuit_feature_cols and {"driverId","circuitId"}.issubset(df.columns):
    if {"season","round"}.issubset(df.columns):
        df_sorted_dc = df.sort_values(["driverId","circuitId","season","round"])
    else:
        df_sorted_dc = df.sort_values(["driverId","circuitId"])
    driver_circuit_feats_ref = (
        df_sorted_dc.groupby(["driverId","circuitId"], as_index=False)[driver_circuit_feature_cols].last()
        .copy()
    )


# Helper to map a user-entered name/code to driverId using drivers.csv
def map_driver_to_id_from_text(s: str):
    if not s or not isinstance(s, str):
        return None
    s_strip = s.strip()
    if BY_CODE and s_strip.upper() in BY_CODE:
        return BY_CODE[s_strip.upper()]
    if BY_NAME and s_strip.lower() in BY_NAME:
        return BY_NAME[s_strip.lower()]
    return None

# If there is no drivers.csv, tell the user we’ll fall back to global means
if BY_CODE is None and BY_NAME is None:
    st.info("Driver mapping not found (data/raw/drivers.csv missing). Driver-specific history will be ignored.")


st.sidebar.header("Settings")

circuit_id_selected = None
round_num_selected = None

if SEASON_CIRCUIT_IDX:
    seasons_sorted = sorted(SEASON_CIRCUIT_IDX.keys(), reverse=True)
    season_sel = st.sidebar.selectbox("Season", options=seasons_sorted, index=0)
    labels = [it["label"] for it in SEASON_CIRCUIT_IDX[season_sel]]
    if labels:
        lab = st.sidebar.selectbox("Circuit (this season)", options=labels, index=0)
        choice = next(it for it in SEASON_CIRCUIT_IDX[season_sel] if it["label"] == lab)
        circuit_id_selected = choice["circuitId"]
        round_num_selected  = choice["round"]
else:
    st.sidebar.info("Couldn’t build season→circuit index (need data/raw/races.csv & circuits.csv).")

mode = st.sidebar.radio("Prediction Mode", ["Single Driver", "Full Grid"])
src = None
if mode == "Full Grid":
    with st.sidebar:
        options = ["Manual table", "Load from local Kaggle CSV", "Upload qualifying CSV", FETCH_LABEL]

        # Default to "Fetch latest qualifying" on first visit.
        prev_src = st.session_state.get("grid_src")
        map_idx = {"manual": 0, "kaggle": 1, "upload": 2, "openf1": 3, "openf1_latest": 3}
        default_idx = map_idx.get(prev_src, options.index(FETCH_LABEL))

        src = st.radio("Grid data source", options, index=default_idx, horizontal=False)

        if src == FETCH_LABEL:
            # Default this ON so “latest” is the normal behavior
            use_latest = st.toggle("Use latest meeting (ignore Season/Round)", value=True, key="openf1_latest")
            fetch_click = st.button("Fetch now", key="fetch_quali_btn")
        else:
            use_latest = False
            fetch_click = False




# ---- Race banner (prefer latest if present) ----
if SEASON_CIRCUIT_IDX and (season_sel is not None) and (round_num_selected is not None):
    use_latest = (st.session_state.get("grid_src") == "openf1_latest")
    meta = st.session_state.get("latest_meta") if use_latest else None

    # Defaults from sidebar
    disp_season  = season_sel
    disp_round   = round_num_selected
    disp_label   = lab  # from your sidebar Season→circuit label
    disp_country = None

    # Override with latest
    if meta:
        disp_season  = meta.get("season", disp_season)
        if meta.get("round") is not None:
            disp_round = meta["round"]
        if meta.get("label"):
            disp_label = meta["label"]              # e.g., "Zandvoort"
        disp_country = meta.get("country", disp_country)

    # Weather for the banner (from your enriched df if we can match)
    t = p = np.nan
    w = 0
    if has_wx and {"season","round"}.issubset(df.columns):
        sel_season = disp_season
        sel_round  = disp_round
        row = df[(df["season"].astype(str) == str(sel_season)) &
                 (df["round"].astype(str)  == str(sel_round))]
        if not row.empty:
            r0 = row.iloc[-1]
            t = r0.get("wx_temp_mean_c", np.nan)
            p = r0.get("wx_precip_mm",   np.nan)
            w = int(r0.get("wx_is_wet", 0) or 0)

    # Country/flag: if latest has a country, DO NOT fall back to sidebar circuit
    country = disp_country
    if not country and not meta and CIRCUITS_DF is not None and (circuit_id_selected is not None) and ("country" in CIRCUITS_DF.columns):
        match = CIRCUITS_DF.loc[CIRCUITS_DF["circuitId"] == circuit_id_selected, "country"]
        if not match.empty:
            country = str(match.iloc[0])

    flag = CIRCUIT_FLAGS.get(str(country), "") if country else ""

    # Build chips
    chips = "".join([
        f'<span class="wx-pill">Temp {t:.1f}°C</span>' if not pd.isna(t) else "",
        f'<span class="wx-pill">Rain {p:.1f} mm</span>' if not pd.isna(p) else "",
        f'<span class="wx-pill">{"Wet" if w else "Dry"}</span>',
    ])

    # Header: show Season + (optional) Round + label
    round_txt = f" — R{int(disp_round):02d}" if disp_round is not None else ""
    header_html = f"""
    <div class="banner">
      <h3>{flag} Season {disp_season}{round_txt} — {disp_label}</h3>
      <div class="chips">{chips}</div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)




#st.success(f"Model trained on {len(df):,} rows. Features: {feature_cols}")

def fetch_quali(season: int, round_num: int) -> pd.DataFrame:
    """
    Return columns: driver, code, grid_quali, grid
    Tries Ergast, then Jolpica (Ergast-compatible), then OpenF1 as a last resort.
    """
    # 1) Ergast + Jolpica (same JSON schema)
    base_urls = [
        f"https://ergast.com/api/f1/{season}/{round_num}/qualifying.json",
        # Jolpica mirror is Ergast-compatible
        f"http://api.jolpi.ca/ergast/f1/{season}/{round_num}/qualifying.json",
    ]
    last_err = None
    for url in base_urls:
        try:
            r = requests.get(url, params={"limit": 1000}, timeout=15)
            r.raise_for_status()
            j = r.json()
            races = j.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not races:
                continue
            rows = []
            for q in races[0].get("QualifyingResults", []):
                drv = q.get("Driver", {})
                code = drv.get("code") or drv.get("familyName") or drv.get("driverId")
                name = f"{drv.get('givenName','')} {drv.get('familyName','')}".strip() or code
                try:
                    qpos = int(q.get("position"))
                except Exception:
                    qpos = None
                if qpos is not None:
                    rows.append({"driver": name, "code": code, "grid_quali": qpos, "grid": qpos})
            if rows:
                return pd.DataFrame(rows).sort_values("grid_quali").reset_index(drop=True)
        except Exception as e:
            last_err = e  # try next mirror

    # 2) OpenF1 (different schema; we map to your columns)
    #    We find the qualifying session_key for the given season/round, then pull the classification.
    try:
        sess = requests.get(
            "https://api.openf1.org/v1/sessions", params={"year": season}, timeout=20
        ).json()

        # Candidate qualifying-like sessions for that year
        cand = [s for s in sess if s.get("session_name") in ("Qualifying", "Sprint Shootout")]

        # Prefer exact round match if OpenF1 exposes 'round'
        q = next((s for s in cand if str(s.get("round", "")) == str(round_num)), None)

        # Fallback heuristic: pick the Nth qualifying of the year
        if q is None and cand:
            cand_sorted = sorted(
                cand, key=lambda s: s.get("date_start") or s.get("session_start_utc") or ""
            )
            idx = max(0, min(int(round_num) - 1, len(cand_sorted) - 1))
            q = cand_sorted[idx]

        if q:
            skey = q["session_key"]
            results = requests.get(
                "https://api.openf1.org/v1/session_result",
                params={"session_key": skey},
                timeout=20,
            ).json()

            rows = []
            for r in results:
                pos = r.get("position")
                if pos is None:
                    continue
                # OpenF1 fields vary across seasons; be defensive
                name = r.get("driver_name") or r.get("broadcast_name") or r.get("full_name")
                code = r.get("driver_code") or str(r.get("driver_number") or "")
                rows.append({"driver": name or code, "code": code, "grid_quali": int(pos), "grid": int(pos)})

            if rows:
                return pd.DataFrame(rows).sort_values("grid_quali").reset_index(drop=True)
    except Exception:
        pass

    # If everything failed, return an empty DF
    return pd.DataFrame(columns=["driver", "code", "grid_quali", "grid"])


# Single driver prediction
if mode == "Single Driver":
    st.subheader("Single Driver: Win Probability")

    with st.form("single_driver_form", clear_on_submit=False):
        c1, c2 = st.columns([1, 2])
        grid = c1.number_input("Starting Grid (1 = pole)", min_value=1, max_value=20, value=1, step=1)
        drv_name = c2.text_input(
            "Driver",
            value="",
            placeholder="e.g., VER, NOR, LEC or full name",
            help="Used to attach driver history & track-affinity features."
        )
        grid_q = None
        if "grid_quali" in feature_cols:
            grid_q = st.number_input("Qualifying Position", min_value=1, max_value=20, value=int(grid), step=1)

        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        st.info("Set inputs above and click **Predict**.")
        st.stop()

    # ✅ BUILD inputs FIRST
    inputs = {"grid": float(grid)}
    if grid_q is not None:
        inputs["grid_quali"] = float(grid_q)

    driver_id_for_input = map_driver_to_id_from_text(drv_name)

    # Driver career features
    if (driver_id_for_input is not None) and (not driver_feats_ref.empty):
        row = driver_feats_ref.loc[driver_feats_ref["driverId"] == driver_id_for_input]
        if not row.empty:
            for c in driver_feature_cols:
                inputs[c] = float(row.iloc[0][c])

    # Driver × circuit features
    if (
        driver_id_for_input is not None
        and circuit_id_selected is not None
        and not driver_circuit_feats_ref.empty
    ):
        row_dc = driver_circuit_feats_ref[
            (driver_circuit_feats_ref["driverId"] == driver_id_for_input) &
            (driver_circuit_feats_ref["circuitId"] == circuit_id_selected)
        ]
        if not row_dc.empty:
            for c in driver_circuit_feature_cols:
                if c in row_dc.columns:
                    inputs[c] = float(row_dc.iloc[0][c])

    # Derived features
    if "grid_sq" in feature_cols:   inputs["grid_sq"]  = inputs["grid"] ** 2
    if "grid_log" in feature_cols:  inputs["grid_log"] = float(np.log1p(inputs["grid"]))
    if "front_row" in feature_cols: inputs["front_row"] = int(inputs["grid"] <= 2)

    if "grid_quali" in inputs:
        gq = float(inputs["grid_quali"])
        if "grid_quali_sq" in feature_cols:   inputs["grid_quali_sq"] = gq ** 2
        if "grid_quali_log" in feature_cols:  inputs["grid_quali_log"] = float(np.log1p(gq))
        if "start_penalty" in feature_cols:   inputs["start_penalty"] = inputs["grid"] - gq

    # Predict
    x_row = np.array([[inputs.get(f, np.nan) for f in feature_cols]], dtype=float)
    row_defaults = feature_defaults.reindex(feature_cols).to_numpy()
    mask = np.isnan(x_row)
    if mask.any():
        x_row[mask] = np.broadcast_to(row_defaults, x_row.shape)[mask]
    x_row = np.nan_to_num(x_row, nan=0.0, posinf=0.0, neginf=0.0)

    prob_win = clf.predict_proba(x_row)[0, 1]
    st.metric("Predicted Win Probability", f"{prob_win*100:.1f}%")
    st.caption("Using the selected features; missing inputs are filled with historical means.")
    
    # --- Top feature contributions (percent of total impact) ---
    contrib_df = explain_logreg_percent(clf, feature_cols, x_row[0]).head(10)
    from streamlit import column_config as _cc
    st.caption("Top feature contributions")
    st.dataframe(
        contrib_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Contribution %": _cc.NumberColumn(format="%.1f%%")
        }
    )


    # --- Top feature contributions (readable names + percent impact) ---

    def _pretty_feature_name(feat: str) -> str:
        base = {
            "grid": "Starting grid (P#)",
            "grid_sq": "Grid²",
            "grid_log": "log(1+Grid)",
            "front_row": "Front row (≤2)",
            "grid_quali": "Qualifying position",
            "grid_quali_sq": "Qualifying²",
            "grid_quali_log": "log(1+Qualifying)",
            "start_penalty": "Grid vs Quali (Δ)",
            "qual_gap_to_pole": "Qualifying gap to pole",
            "track_podium_rate": "Track: podium rate",
            "track_dnf_rate": "Track: DNF rate",
            "wx_temp_mean_c": "Weather: temp (mean, °C)",
            "wx_temp_max_c": "Weather: temp (max, °C)",
            "wx_precip_mm": "Weather: precip (mm)",
            "wx_rain_mm": "Weather: rain (mm)",
            "wx_windspeed_max_kmh": "Weather: wind (max, km/h)",
            "wx_windgusts_max_kmh": "Weather: wind gusts (km/h)",
            "wx_rh_mean_pct": "Weather: humidity (%)",
            "wx_cloud_mean_pct": "Weather: cloud (%)",
            "wx_is_wet": "Weather: wet?",
            "wx_is_hot": "Weather: hot (≥30°C)?",
            "drv_pts_3": "Driver: points (last 3)",
            "team_pts_3": "Team: points (last 3)",
            "drv_pos_3": "Driver: avg finish (last 3)",
            "drv_pts_ytd": "Driver: points YTD",
            "team_pts_ytd": "Team: points YTD",
            "drv_starts_cum": "Driver: career starts",
            "drv_wins_cum": "Driver: career wins",
            "drv_podiums_cum": "Driver: career podiums",
            "drv_points_cum": "Driver: career points",
            "drv_years_exp_so_far": "Driver: years of experience",
            "drv_win_rate_career": "Driver: career win rate",
            "drv_podium_rate_career": "Driver: career podium rate",
            "drv_pts_per_start_career": "Driver: career pts/start",
        }
        if feat in base:
            return base[feat]
        if feat.startswith("drv_track_"):
            tail = feat.replace("drv_track_", "")
            tail_map = {
                "starts_cum": "Driver @track: starts",
                "podiums_cum": "Driver @track: podiums",
                "wins_cum": "Driver @track: wins",
                "podium_rate": "Driver @track: podium rate",
                "win_rate": "Driver @track: win rate",
            }
            return tail_map.get(tail, f"Driver @track: {tail.replace('_',' ')}")
        if feat.startswith("drv_"):
            return "Driver: " + feat.replace("drv_", "").replace("_", " ")
        if feat.startswith("team_"):
            return "Team: " + feat.replace("team_", "").replace("_", " ")
        if feat.startswith("wx_"):
            return "Weather: " + feat.replace("wx_", "").replace("_", " ")
        # fallback
        return feat.replace("_", " ").title()

    def _pretty_value(feat: str, val: float) -> str:
        try:
            v = float(val)
        except Exception:
            return str(val)
        if feat in {"front_row", "wx_is_wet", "wx_is_hot"}:
            return "Yes" if v >= 0.5 else "No"
        if "temp_" in feat:
            return f"{v:.1f} °C"
        if "wind" in feat:
            return f"{v:.0f} km/h"
        if "precip" in feat or "rain" in feat:
            return f"{v:.1f} mm"
        if "rate" in feat:
            # heuristic: many *_rate features are already 0–1
            return f"{(v*100):.1f}%" if 0 <= v <= 1 else f"{v:.2f}"
        if v.is_integer():
            return f"{int(v)}"
        return f"{v:.2f}"

    # Compute per-feature log-odds contributions for THIS prediction
    coefs = clf.coef_[0]
    xvec = x_row[0]  # from the Single-Driver computation above
    raw = pd.DataFrame({
        "feature": feature_cols,
        "coef": coefs,
        "value": xvec,
    })
    raw["contrib"] = raw["coef"] * raw["value"]  # contribution to log-odds
    raw["abs_contrib"] = raw["contrib"].abs()

    abs_total = float(raw["abs_contrib"].sum())
    if abs_total == 0:
        abs_total = 1e-9  # avoid divide-by-zero

    raw["impact_pct"] = (raw["abs_contrib"] / abs_total) * 100.0
    raw["direction"] = np.where(raw["contrib"] >= 0, "↑ helps", "↓ hurts")
    raw["Feature"] = raw["feature"].apply(_pretty_feature_name)
    raw["Value"] = [ _pretty_value(f, v) for f, v in zip(raw["feature"], raw["value"]) ]

    # Show top-K most influential (by absolute impact)
    K = 10
    top = (raw.sort_values("abs_contrib", ascending=False)
            .head(K)
            .loc[:, ["Feature", "Value", "impact_pct", "direction"]]
            .rename(columns={"impact_pct": "Impact %", "direction": "Effect"}))

    from streamlit import column_config
    st.caption("Top feature contributions (share of model impact for this prediction)")
    st.dataframe(
        top,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Feature": column_config.TextColumn("Feature"),
            "Value": column_config.TextColumn("Current value"),
            "Impact %": column_config.ProgressColumn("Impact", format="%.1f%%", min_value=0.0, max_value=100.0),
            "Effect": column_config.TextColumn("Effect"),
        },
    )
    st.caption("Impact % is each feature’s share of the model’s total absolute log-odds impact; ↑/↓ shows whether it increases or decreases the win chances for this input.")



    
# Full grid prediction
elif mode == "Full Grid":
    tab_pred, tab_fant, tab_model = st.tabs(["🔮 Predictions", "🎮 Fantasy", "📈 Model quality"])

    # --- Session init for persistence across reruns ---
    if "grid_df" not in st.session_state:
        st.session_state.grid_df = None
    if "grid_src" not in st.session_state:
        st.session_state.grid_src = None
    if "last_ctx" not in st.session_state:
        st.session_state.last_ctx = (season_sel, round_num_selected)
        
    if "did_autoload_latest" not in st.session_state:
        st.session_state.did_autoload_latest = False

    # Auto-load latest quali exactly once on first visit to Full Grid
    if st.session_state.grid_df is None and not st.session_state.did_autoload_latest:
        with st.spinner("Fetching latest qualifying (OpenF1)…"):
            dfq = fetch_latest_quali_openf1()
            if not dfq.empty:
                st.session_state.grid_df = dfq
                st.session_state.grid_src = "openf1_latest"
                user_df = dfq
                st.toast(f"Loaded {len(dfq)} quali rows (latest meeting)", icon="✅")
            else:
                st.warning("No latest qualifying found yet.")
            st.session_state.did_autoload_latest = True

    # Reset auto-loaded grids when season/round changes
    if (season_sel, round_num_selected) != st.session_state.last_ctx:
        # Only reset for sources that are tied to a specific round
        if st.session_state.grid_src in ("kaggle", "ergast", "openf1"):
            st.session_state.grid_df = None
            st.session_state.grid_src = None
            st.session_state.did_autoload_latest = False  # allow a fresh autoload if needed
        st.session_state.last_ctx = (season_sel, round_num_selected)


    # Start from session
    user_df = st.session_state.grid_df
    
    # Handle each source
    # --- OpenF1 fetch (selected round or latest) ---
    if src == FETCH_LABEL and fetch_click:
        with st.spinner("Fetching from OpenF1…"):
            if use_latest:
                dfq = fetch_latest_quali_openf1()
            else:
                if SEASON_CIRCUIT_IDX and season_sel is not None and round_num_selected is not None:
                    dfq = fetch_quali_openf1(int(season_sel), int(round_num_selected))
                else:
                    st.warning("Pick a Season and Circuit in the sidebar first.")
                    dfq = pd.DataFrame()

            if dfq.empty:
                st.warning("No qualifying data found yet.")
            else:
                st.toast(f"Loaded {len(dfq)} quali rows", icon="✅")
                user_df = dfq
                st.session_state.grid_df = dfq
                st.session_state.grid_src = "openf1_latest" if use_latest else "openf1"


    elif src == "Load from local Kaggle CSV":
    # Auto-load immediately when Season/Circuit is chosen (no button needed)
        if SEASON_CIRCUIT_IDX and season_sel is not None and round_num_selected is not None:
            # Re-load if there's nothing yet, or if previous source wasn't Kaggle
            need_autoload = (st.session_state.grid_df is None) or (st.session_state.grid_src != "kaggle")
            if need_autoload:
                try:
                    auto_df = load_local_quali(int(season_sel), int(round_num_selected))
                    if auto_df.empty:
                        st.warning("No qualifying rows found for that season/circuit. Check your Kaggle coverage.")
                    else:
                        st.session_state.grid_df = auto_df
                        st.session_state.grid_src = "kaggle"
                        user_df = auto_df
                        st.toast(f"Loaded {len(auto_df)} entries from data/raw/*.csv", icon="✅")

                        # (Optional) small weather snippet using your enriched file
                        if has_wx and "season" in df.columns and "round" in df.columns:
                            cand = df[
                                (df["season"].astype(str) == str(season_sel)) &
                                (df["round"].astype(str)  == str(round_num_selected))
                            ].dropna(subset=[c for c in wx_cols if c in df.columns], how="all")
                            if not cand.empty:
                                r0 = cand.iloc[0]
                                st.caption(
                                    f"Weather (from enriched data): "
                                    f"mean {r0.get('wx_temp_mean_c', float('nan')):.1f}°C, "
                                    f"max {r0.get('wx_temp_max_c', float('nan')):.1f}°C, "
                                    f"precip {r0.get('wx_precip_mm', float('nan')):.1f} mm, "
                                    f"{'wet' if int(r0.get('wx_is_wet', 0) or 0) else 'dry'}"
                                )
                except Exception as e:
                    st.error(f"Local load failed: {e}")
                    st.info("Ensure data/raw/races.csv, qualifying.csv, and drivers.csv exist (from your Kaggle download).")
            else:
                # Already have Kaggle data in session; just use it
                user_df = st.session_state.grid_df
        else:
            st.warning("Pick a Season and Circuit in the sidebar first.")


    elif src == "Upload qualifying CSV":
        st.write("CSV should include **driver name** and **qualifying position**.")
        up = st.file_uploader("Upload qualifying CSV", type=["csv"])
        if up is not None:
            qcsv = pd.read_csv(up)
            name_col = next((c for c in ["driver","name","surname","familyName","code","Driver"] if c in qcsv.columns), None)
            pos_col  = next((c for c in ["grid_quali","position","qualifying_position","qpos","P"] if c in qcsv.columns), None)
            if name_col and pos_col:
                tmp = qcsv[[name_col, pos_col]].rename(columns={name_col:"driver", pos_col:"grid_quali"}).copy()
                tmp["grid"] = pd.to_numeric(tmp["grid_quali"], errors="coerce")
                tmp = tmp.dropna(subset=["grid"]).sort_values("grid").reset_index(drop=True)
                user_df = tmp
                st.session_state.grid_df = tmp
                st.session_state.grid_src = "upload"
                st.success(f"Parsed {len(user_df)} rows from uploaded file.")
            else:
                st.error("Could not detect name/position columns. Include columns like 'driver' and 'position' (or 'grid_quali').")

    # Default manual if nothing yet
    if user_df is None:
        n_default = 10
        data = {"driver": [f"Driver {i}" for i in range(1, n_default+1)],
                "grid": list(range(1, n_default+1))}
        if "grid_quali" in feature_cols:
            data["grid_quali"] = list(range(1, n_default+1))
        user_df = pd.DataFrame(data)
        st.session_state.grid_df = user_df
        st.session_state.grid_src = "manual"

    # --- Show editor only for Manual table; otherwise skip straight to predictions ---
    show_editor = (src == "Manual table") or (st.session_state.get("grid_src") == "manual")

    if show_editor:
        editor_key = "grid_editor_fullgrid"
        edit_df = st.data_editor(
            user_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "driver": column_config.TextColumn("Driver", help="3-letter code or full name"),
                "grid": column_config.NumberColumn("Grid", min_value=1, max_value=20, step=1),
                **({"grid_quali": column_config.NumberColumn("Quali", min_value=1, max_value=20, step=1)}
                if "grid_quali" in user_df.columns else {}),
            },
            key=editor_key,
        )
        # Persist edits
        st.session_state.grid_df = edit_df
    else:
        # Kaggle / OpenF1 / Upload: no editor, go straight to predictions
        edit_df = user_df.copy()


    # --- Clean input ---
    edit_df["grid"] = pd.to_numeric(edit_df.get("grid"), errors="coerce")
    if "grid_quali" in edit_df.columns:
        edit_df["grid_quali"] = pd.to_numeric(edit_df["grid_quali"], errors="coerce")

    pred_rows = edit_df[edit_df["grid"].notna()].copy()
    if pred_rows.empty:
        st.info("No valid rows to predict (need at least a grid value).")
        st.stop()

    # --- Derived features ---
    if "grid_sq" in feature_cols:   pred_rows["grid_sq"]  = pred_rows["grid"].astype(float) ** 2
    if "grid_log" in feature_cols:  pred_rows["grid_log"] = np.log1p(pred_rows["grid"].astype(float))
    if "front_row" in feature_cols: pred_rows["front_row"] = (pred_rows["grid"].astype(float) <= 2).astype(int)
    if "grid_quali" in pred_rows:
        if "grid_quali_sq" in feature_cols:  pred_rows["grid_quali_sq"]  = pred_rows["grid_quali"].astype(float) ** 2
        if "grid_quali_log" in feature_cols: pred_rows["grid_quali_log"] = np.log1p(pred_rows["grid_quali"].astype(float))
        if "start_penalty" in feature_cols:  pred_rows["start_penalty"]  = pred_rows["grid"].astype(float) - pred_rows["grid_quali"].astype(float)
    
    # --- Seed race-level weather into pred_rows so the model can use them ---
    # Uses t (mean temp) and w (wet flag) computed above for the selected season/round.
    if has_wx:
        # Only create columns that the model actually trained with
        if "wx_is_wet" in feature_cols:
            pred_rows["wx_is_wet"] = int(w) if not pd.isna(w) else 0

        if "wx_is_hot" in feature_cols:
            # Prefer explicit hot flag if you have it; otherwise, derive from temp >= 30°C
            base_hot = 1 if (not pd.isna(t) and float(t) >= 30.0) else 0
            pred_rows["wx_is_hot"] = base_hot

        # (Optional) seed continuous weather features if your model uses them
        for col, src in [
            ("wx_temp_mean_c", t),
            ("wx_precip_mm",   p),
        ]:
            if col in feature_cols and not pd.isna(src):
                pred_rows[col] = float(src)


    # --- Attach driver/circuit features ---
    if (BY_CODE or BY_NAME):
        pred_rows["driverId"] = pred_rows.apply(
            lambda r: map_driver_to_id_from_row(r.get("driver"), r.get("code")),
            axis=1
        )
        pred_rows["driverId"] = pd.to_numeric(pred_rows["driverId"], errors="coerce").astype("Int64")

    # ✅ Make sure keys are the same dtype before merging
    if "driverId" in pred_rows.columns:
        pred_rows = _force_int64_key(pred_rows, ["driverId"])

    if not driver_feats_ref.empty and "driverId" in driver_feats_ref.columns:
        driver_feats_ref = _force_int64_key(driver_feats_ref, ["driverId"])

    if "driverId" in pred_rows.columns and not driver_feats_ref.empty:
        pred_rows = pred_rows.merge(driver_feats_ref, on="driverId", how="left")

    # Keep circuitId if you have it
    if circuit_id_selected is not None:
        pred_rows["circuitId"] = int(circuit_id_selected)

    # ✅ For the driver×circuit merge, coerce BOTH keys on BOTH frames
    if {"driverId","circuitId"}.issubset(pred_rows.columns) and not driver_circuit_feats_ref.empty:
        pred_rows                = _force_int64_key(pred_rows, ["driverId","circuitId"])
        driver_circuit_feats_ref = _force_int64_key(driver_circuit_feats_ref, ["driverId","circuitId"])
        pred_rows = pred_rows.merge(driver_circuit_feats_ref, on=["driverId","circuitId"], how="left")
    
    # --- Constructors / team mapping for Fantasy builder ---
    try:
        cons_map = build_latest_constructor_map(df)   # from your processed history
        if "driverId" in pred_rows.columns:
            # Make dtypes compatible (avoid object vs Int64 merge errors)
            pred_rows["driverId"] = pd.to_numeric(pred_rows["driverId"], errors="coerce").astype("Int64")
        if cons_map is not None and not cons_map.empty and "driverId" in pred_rows.columns:
            cons_map = cons_map.copy()
            cons_map["driverId"] = pd.to_numeric(cons_map["driverId"], errors="coerce").astype("Int64")

            # Merge but DO NOT lose any 'team' already present from OpenF1
            pred_rows = pred_rows.merge(cons_map, on="driverId", how="left", suffixes=("", "_hist"))

            # If we didn’t have a team from OpenF1, use the historical one
            if "team" not in pred_rows.columns:
                pred_rows["team"] = pred_rows.get("team_hist")
            else:
                pred_rows["team"] = pred_rows["team"].fillna(pred_rows.get("team_hist"))
            # Clean up
            drop_cols = [c for c in ["team_hist"] if c in pred_rows.columns]
            if drop_cols:
                pred_rows.drop(columns=drop_cols, inplace=True)

        # Final safety: ensure 'team' column exists for the Team Builder
        if "team" not in pred_rows.columns:
            pred_rows["team"] = None
        pred_rows["team"] = pred_rows["team"].fillna("").astype(str)

    except Exception as e:
        st.warning(f"Constructor map failed: {e}")
        if "team" not in pred_rows.columns:
            pred_rows["team"] = ""


        
    # Fallback: if team is still missing but we have constructorId, map via data/raw/constructors.csv
    if (("team" not in pred_rows.columns) or pred_rows["team"].isna().all()) and ("constructorId" in pred_rows.columns):
        cp = Path("data/raw/constructors.csv")
        if cp.exists():
            con = pd.read_csv(cp)
            lc = {c.lower(): c for c in con.columns}
            cid = lc.get("constructorid", "constructorId")
            cname = lc.get("name", "name")
            if cid in con.columns and cname in con.columns:
                pred_rows = pred_rows.merge(
                    con[[cid, cname]].rename(columns={cid: "constructorId", cname: "team"}),
                    on="constructorId", how="left"
                )
                pred_rows["team"] = pred_rows["team"].fillna("").astype(str)


    
    # --- BASE matrix & predict (race-level softmax) ---
    X_base_df = pred_rows.reindex(columns=feature_cols).fillna(feature_defaults).fillna(0.0)
    X_base = np.nan_to_num(X_base_df.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    # Pull tuned params if available, else defaults
    comb = st.session_state.get("best_combiner") or {}
    TEMP_T  = float(comb.get("temp_T", 1.25))
    BLEND   = float(comb.get("blend", 0.6))
    G_ALPHA = float(comb.get("grid_alpha", 0.08))
    G_BETA  = float(comb.get("gap_beta", 0.06))

    proba_base = compute_win_probs(
        X_base, clf=clf,
        grid=pred_rows.get("grid"),
        start_penalty=(pred_rows["grid"] - pred_rows["grid_quali"]) if {"grid","grid_quali"}.issubset(pred_rows.columns) else None,
        method="blend",
        grid_alpha=G_ALPHA, gap_beta=G_BETA, temp_T=TEMP_T, blend=BLEND,
    )

    with tab_pred:
        # Subheader lives INSIDE the Predict tab so it won't appear under every tab
        st.subheader("Full Grid: Predict Winner Among Multiple Drivers")
        st.markdown(st.session_state.get("race_banner_html",""), unsafe_allow_html=True)


        # Build + sort base table
        table_df = pred_rows.copy()
        table_df["win_prob"] = proba_base
        table_df = table_df.sort_values("win_prob", ascending=False).reset_index(drop=True)
        table_df["win_pct"] = table_df["win_prob"] * 100.0

        # --- INIT (once) and READ what-if state ---
        if "whatif_delta_grid" not in st.session_state:
            st.session_state["whatif_delta_grid"] = 0
        if "whatif_wet" not in st.session_state:
            st.session_state["whatif_wet"] = (bool(w) if 'w' in locals() and not pd.isna(w) else False)
        if "whatif_hot" not in st.session_state:
            st.session_state["whatif_hot"] = ((float(t) >= 30.0) if 't' in locals() and not pd.isna(t) else False)

        dgrid    = st.session_state["whatif_delta_grid"]
        wet_flag = st.session_state["whatif_wet"]
        hot_flag = st.session_state["whatif_hot"]

        # --- ALWAYS APPLY what-if tweaks to the table ---
        tuned = table_df.copy()

        # grid tweaks + derived features
        if "grid" in tuned.columns:
            tuned["grid"] = np.clip((tuned["grid"] + dgrid).fillna(tuned["grid"]), 1, 20)
            if "grid_sq" in feature_cols:   tuned["grid_sq"]  = tuned["grid"].astype(float) ** 2
            if "grid_log" in feature_cols:  tuned["grid_log"] = np.log1p(tuned["grid"].astype(float))
            if "front_row" in feature_cols: tuned["front_row"] = (tuned["grid"].astype(float) <= 2).astype(int)

        # weather toggles: force-create and override if model uses them
        if "wx_is_wet" in feature_cols:
            tuned["wx_is_wet"] = 1 if wet_flag else 0
        if "wx_is_hot" in feature_cols:
            tuned["wx_is_hot"] = 1 if hot_flag else 0

        # keep start_penalty consistent when grid shifts
        if "start_penalty" in feature_cols and "grid_quali" in tuned.columns:
            tuned["start_penalty"] = tuned["grid"].astype(float) - tuned["grid_quali"].astype(float)

        # re-predict with tweaks
        X_tuned_df = tuned.reindex(columns=feature_cols).fillna(feature_defaults).fillna(0.0)
        X_tuned    = np.nan_to_num(X_tuned_df.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        tuned["win_prob"] = compute_win_probs(
            X_tuned, clf=clf,
            grid=tuned.get("grid"),
            start_penalty=(tuned["grid"] - tuned["grid_quali"]) if {"grid","grid_quali"}.issubset(tuned.columns) else None,
            method="blend",
            grid_alpha=G_ALPHA, gap_beta=G_BETA, temp_T=TEMP_T, blend=BLEND,
        )

        tuned = tuned.sort_values("win_prob", ascending=False).reset_index(drop=True)
        tuned["win_pct"] = tuned["win_prob"] * 100.0

        # final table used everywhere (Fantasy reads this)
        table_df = tuned

        # ⇩ Add rank/medals column
        rank = np.arange(len(table_df)) + 1
        medals = np.where(rank==1, "🥇", np.where(rank==2, "🥈", np.where(rank==3, "🥉", rank.astype(str))))
        table_df.insert(0, "Rank", medals)

        # --- TABLE FIRST  ---
        display_df = table_df.copy()
        show_cols = ["Rank"] + [c for c in ["driver","grid","grid_quali","wx_temp_max_c"] if c in display_df.columns] + ["win_pct"]
        display_cols = [c for c in show_cols if c in display_df.columns]

        st.header("Predicted Win Probabilities")
        comb = st.session_state.get("best_combiner")
        if comb:
            try:
                top1_txt = f"{comb.get('top1', float('nan'))*100:.0f}%" if comb.get('top1') is not None else "—"
            except Exception:
                top1_txt = "—"
            st.caption(
                f"Auto-tuned parameters active: T={comb.get('temp_T',1.25)}, "
                f"blend={comb.get('blend',0.6)}, "
                f"α={comb.get('grid_alpha',0.08)}, β={comb.get('gap_beta',0.06)} · "
                f"Top-1≈{top1_txt}"
            )

        st.dataframe(
            display_df[display_cols],
            use_container_width=True,
            height=420,
            column_config={
                **({"grid": column_config.NumberColumn("Grid", format="%d")} if "grid" in display_cols else {}),
                **({"grid_quali": column_config.NumberColumn("Quali", format="%d")} if "grid_quali" in display_cols else {}),
                "win_pct": column_config.ProgressColumn("Win %", format="%.1f%%", min_value=0.0, max_value=100.0),
            },
            hide_index=True,
        )

        # === WHAT-IF CONTROLS (MOVED HERE) ===
        st.markdown("""
        <style>
        .whatif-title { font-size: 1.5rem; font-weight: 800; letter-spacing: .2px; margin: 6px 0 4px; line-height: 1.3; }
        .biglabel    { font-size: 1.08rem; font-weight: 700; margin: 6px 0 4px; line-height: 1.35; }
        .subhead     { font-size: 1.28rem; font-weight: 700; margin: 10px 0 4px; }
        </style>
        <div class="whatif-title">🔧 Tweak inputs</div>
        """, unsafe_allow_html=True)

        with st.expander("", expanded=True):
            st.markdown('<div class="biglabel">Grid adjustment (±)</div>', unsafe_allow_html=True)
            st.slider("", -3, 3, dgrid, 1, key="whatif_delta_grid", label_visibility="collapsed")

            st.markdown('<div class="subhead">Race conditions</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="biglabel">Wet race</div>', unsafe_allow_html=True)
                st.toggle("", value=wet_flag, key="whatif_wet", label_visibility="collapsed")
            with c2:
                st.markdown('<div class="biglabel">Hot (≥30°C)</div>', unsafe_allow_html=True)
                st.toggle("", value=hot_flag, key="whatif_hot", label_visibility="collapsed")

            st.caption("Changes update the table above automatically.")
            if st.button("Reset tweaks", type="secondary"):
                base_wet = bool(w) if ('w' in locals() and not pd.isna(w)) else False
                base_hot = (float(t) >= 30.0) if ('t' in locals() and not pd.isna(t)) else False
                st.session_state["whatif_delta_grid"] = 0
                st.session_state["whatif_wet"] = base_wet
                st.session_state["whatif_hot"] = base_hot
                st.rerun()

        # --- Winner + download (MOVED HERE) ---
        top_row = table_df.iloc[0]
        winner_name = top_row["driver"] if "driver" in table_df.columns and pd.notna(top_row["driver"]) else f"Grid {int(top_row['grid'])}"
        st.success(f"🏆 Predicted winner: **{winner_name}**  (Win prob: {top_row['win_prob'] * 100:.1f}%)")

        st.download_button(
            "⬇️ Download predictions (CSV)",
            table_df[display_cols + (["win_prob"] if "win_prob" not in display_cols else [])]
                .to_csv(index=False).encode(),
            file_name=(
                f"f1_predictions_s{season_sel}_r{round_num_selected}.csv"
                if (SEASON_CIRCUIT_IDX and season_sel is not None and round_num_selected is not None)
                else "f1_predictions.csv"
            ),
            mime="text/csv",
        )

    with tab_fant: 
        st.markdown(st.session_state.get("race_banner_html",""), unsafe_allow_html=True)
        # --- 🎮 Fantasy helper: Expected points & DRS recommendation ---
        st.header("🎮 Fantasy: Expected Points & DRS Boost")

        # Advanced settings (collapse by default)
        with st.expander("Advanced settings", expanded=False):
            pts_str = st.text_input(
                "Finish points for P1..P10 (comma-sep)",
                value="25,18,15,12,10,8,6,4,2,1",
                help="Edit to match your fantasy scoring."
            )
            n_sims = st.slider("Number of simulations", 1000, 30000, 8000, step=1000)
            seed   = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)

        # Defaults if expander stays closed
        if "pts_str" not in locals(): pts_str = "25,18,15,12,10,8,6,4,2,1"
        if "n_sims" not in locals(): n_sims = 8000
        if "seed"   not in locals(): seed   = 42

        # DRS: ×2 by default; ×3 automatically when Wildcard is active
        wildcard = st.toggle("Wildcard active (DRS ×3)", value=False)
        drs_mult = 3.0 if wildcard else 2.0

        # Parse finish points safely
        try:
            FINISH_POINTS = np.array([float(x.strip()) for x in pts_str.split(",") if x.strip() != ""], dtype=float)
        except Exception:
            FINISH_POINTS = np.array([25,18,15,12,10,8,6,4,2,1], dtype=float)

        # Run sims using your current table's win probabilities
        win_vec = table_df["win_prob"].to_numpy(copy=True)
        k_for_points = min(20, FINISH_POINTS.size)
        order, exp_pos, pos_counts, pos_rate = simulate_topk_from_winprob(
            win_vec, n_sims=int(n_sims), seed=int(seed), k=int(k_for_points)
        )

        # Expected finish points from position probabilities
        P = pos_rate / 100.0
        max_k = min(P.shape[1], FINISH_POINTS.size)
        exp_finish_pts = (P[:, :max_k] * FINISH_POINTS[:max_k]).sum(axis=1)

        fantasy_df = table_df.copy()
        fantasy_df["exp_points"] = exp_finish_pts
        fantasy_df["drs_gain"]   = (float(drs_mult) - 1.0) * fantasy_df["exp_points"]
        fantasy_df = fantasy_df.sort_values(["exp_points","win_prob"], ascending=[False, False]).reset_index(drop=True)
        
        from streamlit import column_config as _cc
        st.dataframe(
            fantasy_df[[c for c in ["driver","grid","grid_quali","win_pct","exp_points","drs_gain"] if c in fantasy_df.columns]],
            use_container_width=True, hide_index=True,
            column_config={
                **({"grid": _cc.NumberColumn("Grid", format="%d")} if "grid" in fantasy_df.columns else {}),
                **({"grid_quali": _cc.NumberColumn("Quali", format="%d")} if "grid_quali" in fantasy_df.columns else {}),
                "win_pct":     _cc.ProgressColumn("Win %",      min_value=0.0, max_value=100.0, format="%.1f%%"),
                "exp_points":  _cc.NumberColumn("Exp. points",  format="%.2f"),
                "drs_gain":    _cc.NumberColumn("DRS gain (Δ)", format="%.2f"),
            }
        )

        # Recommendation callout
        best_idx = int(np.argmax(fantasy_df["drs_gain"].to_numpy()))
        best_row = fantasy_df.iloc[best_idx]
        st.success(f"💡 DRS recommendation: **{best_row.get('driver','(unknown)')}** "
                f"(+{best_row['drs_gain']:.2f} expected points at ×{drs_mult:.1f})")
        
        # === 🧩 Team Builder: pick Drivers + Constructor in one place ===
        st.subheader("Team Builder")

        with st.expander("Build your team", expanded=True):
            # Make sure we carry a usable team string into the pool
            if "team" not in fantasy_df.columns:
                # try to carry it over from pred_rows if it exists
                fantasy_df["team"] = pred_rows.get("team") if "pred_rows" in locals() else ""
            fantasy_df["team"] = fantasy_df["team"].fillna("").astype(str)
            if show_dbg:
                st.caption(f"Teams detected in pred_rows: {sorted(set([t for t in pred_rows.get('team', []) if isinstance(t, str) and t]))}")

            cons_base = (
                fantasy_df.loc[fantasy_df["team"] != "", ["team", "exp_points"]]
                        .groupby("team", as_index=False)["exp_points"].sum()
                        .rename(columns={"team": "constructor"})
            )

            if cons_base.empty:
                st.info("No team mapping for drivers yet — cannot build constructor table.")
            else:
                # (everything that was under your old 'else:' stays the same)
                cons_base = cons_base.sort_values("exp_points", ascending=False).drop_duplicates("constructor")

                c_left, c_right = st.columns([1.35, 1.0])

                with c_left:
                    st.markdown("**Drivers pool**")
                    drv_cols = [c for c in ["driver","team","grid","win_pct","exp_points","price"] if c in fantasy_df.columns]
                    for flag in ("lock","exclude"):
                        if flag not in fantasy_df.columns:
                            fantasy_df[flag] = False

                    drv_edit = st.data_editor(
                        fantasy_df[drv_cols + ["lock","exclude"]],
                        use_container_width=True, hide_index=True, key="tb_driver_editor",
                        column_config={
                            "win_pct": column_config.NumberColumn("Win %", format="%.1f"),
                            "exp_points": column_config.NumberColumn("Exp. points", format="%.2f"),
                            "price": column_config.NumberColumn("Price", format="%.2f"),
                            "lock": column_config.CheckboxColumn("Lock"),
                            "exclude": column_config.CheckboxColumn("Exclude"),
                        }
                    )

                with c_right:
                    st.markdown("**Constructors pool**")
                    cons_cols = ["constructor","exp_points"]
                    if "price" not in cons_base.columns:
                        cons_base["price"] = 0.0
                    cons_edit = st.data_editor(
                        cons_base[cons_cols + ["price"]],
                        use_container_width=True, hide_index=True, key="tb_cons_editor",
                        column_config={
                            "exp_points": column_config.NumberColumn("Exp. points", format="%.2f"),
                            "price": column_config.NumberColumn("Price", format="%.2f"),
                        }
                    )

                st.markdown("---")

                c1, c2, c3, c4 = st.columns([1,1,1,1.2])
                with c1:
                    n_drivers = st.number_input("Drivers", 1, 6, 5, step=1, key="tb_n_drivers")
                with c2:
                    n_cons = st.number_input("Constructors", 1, 2, 1, step=1, key="tb_n_cons")
                with c3:
                    use_budget = st.toggle("Use budget", value=False, key="tb_use_budget")
                with c4:
                    budget = st.number_input("Budget", 0.0, 1000.0, 100.0, step=1.0, key="tb_budget", disabled=not use_budget)

                def _safe_sort(df, by_desc):
                    cols = [c for c in by_desc if c in df.columns]
                    if not cols:
                        return df
                    return df.sort_values(cols, ascending=[False]*len(cols))

                locked_drv = drv_edit[drv_edit.get("lock", False) == True].copy()
                excl_drv   = drv_edit[drv_edit.get("exclude", False) == True].copy()
                pool_drv   = drv_edit.loc[~drv_edit.index.isin(locked_drv.index) & ~drv_edit.index.isin(excl_drv.index)].copy()

                pick_drv = locked_drv.copy()
                pick_cons = pd.DataFrame(columns=["constructor","exp_points","price"])

                def _sum_price(df): return float(df["price"].sum()) if "price" in df.columns else 0.0
                spend = _sum_price(pick_drv)
                cap   = float(budget) if use_budget else float("inf")

                need_drv = max(0, int(n_drivers) - len(pick_drv))
                pool_drv = _safe_sort(pool_drv, ["exp_points","win_pct","price"])
                for _, r in pool_drv.iterrows():
                    if need_drv == 0: break
                    cost = float(r["price"]) if "price" in r else 0.0
                    if spend + cost <= cap:
                        pick_drv = pd.concat([pick_drv, pd.DataFrame([r])], ignore_index=True)
                        spend += cost
                        need_drv -= 1

                cons_pool = _safe_sort(cons_edit.copy(), ["exp_points","price"])
                need_cons = int(n_cons)
                for _, r in cons_pool.iterrows():
                    if need_cons == 0: break
                    cost = float(r["price"]) if "price" in r else 0.0
                    if spend + cost <= cap:
                        pick_cons = pd.concat([pick_cons, pd.DataFrame([r])], ignore_index=True)
                        spend += cost
                        need_cons -= 1

                if len(pick_drv) < int(n_drivers):
                    pick_drv = _safe_sort(drv_edit, ["exp_points","win_pct"]).head(int(n_drivers)).copy()
                if len(pick_cons) < int(n_cons):
                    pick_cons = _safe_sort(cons_edit, ["exp_points"]).head(int(n_cons)).copy()

                if not pick_drv.empty and float(drs_mult) > 1.0:
                    idx = int(pick_drv["exp_points"].idxmax())
                    pick_drv.loc[idx, "exp_points"] = float(pick_drv.loc[idx, "exp_points"]) * float(drs_mult)

                st.markdown("**Recommended team**")
                cc1, cc2 = st.columns([1.35, 1.0])
                with cc1:
                    st.caption("Drivers")
                    show_cols_d = [c for c in ["driver","team","grid","win_pct","exp_points","price"] if c in pick_drv.columns]
                    st.dataframe(
                        pick_drv[show_cols_d].reset_index(drop=True),
                        use_container_width=True, hide_index=True,
                        column_config={
                            "win_pct": column_config.NumberColumn("Win %", format="%.1f"),
                            "exp_points": column_config.NumberColumn("Exp. points", format="%.2f"),
                            "price": column_config.NumberColumn("Price", format="%.2f"),
                        }
                    )
                with cc2:
                    st.caption("Constructor(s)")
                    show_cols_c = [c for c in ["constructor","exp_points","price"] if c in pick_cons.columns]
                    st.dataframe(
                        pick_cons[show_cols_c].reset_index(drop=True),
                        use_container_width=True, hide_index=True,
                        column_config={
                            "exp_points": column_config.NumberColumn("Exp. points", format="%.2f"),
                            "price": column_config.NumberColumn("Price", format="%.2f"),
                        }
                    )

                total_points = float(pick_drv["exp_points"].sum()) + float(pick_cons["exp_points"].sum())
                total_price  = _sum_price(pick_drv) + _sum_price(pick_cons)
                st.success(f"Estimated total points: **{total_points:.2f}** · Spend: **{total_price:.2f}**")

        
        

    with tab_model:
        # === Simple Model Quality check (stateful & no nested expanders) ===
        with st.expander("How good is this model? (one-click check)", expanded=False):
            st.caption("We replay past seasons we didn't train on and see how often our top pick actually won.")

            # Init session state buckets
            if "modelq_overall" not in st.session_state:
                st.session_state.modelq_overall = None
                st.session_state.modelq_per_season = None
                st.session_state.modelq_error = None

            # Run backtest on click -> save results to session_state
            run_now  = st.button("Run model check", type="primary", key="modelq_run")
            if run_now:
                if not {"season", "raceId"}.issubset(df.columns):
                    st.session_state.modelq_error = "Needs 'season' and 'raceId' columns in your processed file."
                    st.session_state.modelq_overall = None
                    st.session_state.modelq_per_season = None
                else:
                    with st.spinner("Running backtest by season…"):
                        try:
                            per_season, overall = backtest_by_season(
                                df[["season","raceId","win", *feature_cols]].copy(),
                                feature_cols,
                                calibrate=True
                            )
                            st.session_state.modelq_overall = overall
                            st.session_state.modelq_per_season = per_season
                            st.session_state.modelq_error = None
                        except Exception as e:
                            st.session_state.modelq_error = str(e)
                            st.session_state.modelq_overall = None
                            st.session_state.modelq_per_season = None
                            
            tune_now = st.button("Improve Probabilities (auto-tune)", key="modelq_tune")
            if tune_now:
                with st.spinner("Searching combiner params over past seasons…"):
                    try:
                        # Build a de-duplicated list of columns for the tuner
                        cols_for_tune = pd.Index(["season", "raceId", "win", *feature_cols]).drop_duplicates()

                        best = autotune_combiner(
                            df.loc[:, cols_for_tune].copy(),
                            feature_cols=list(pd.Index(feature_cols).drop_duplicates())
                        )
                        st.session_state["best_combiner"] = best
                        if best:
                            st.success(
                                f"Best params → T={best['temp_T']}, blend={best['blend']}, "
                                f"grid_alpha={best['grid_alpha']}, gap_beta={best['gap_beta']} "
                                f"(Top-1 ~ {best['top1']*100:.0f}%, NLL {best['score_nll']:.3f})"
                            )
                    except Exception as e:
                        st.error(f"Tuning failed: {e}")

            # Render (persists across reruns so the toggle works)
            if st.session_state.modelq_error:
                st.error(f"Backtest failed: {st.session_state.modelq_error}")
            elif st.session_state.modelq_overall is not None:
                overall    = st.session_state.modelq_overall
                per_season = st.session_state.modelq_per_season

                top1  = float(overall.get("top1_hit_overall") or 0.0)
                brier = overall.get("brier_weighted")
                brier = float(brier) if brier is not None else float("nan")

                def letter_grade(top1_hit: float, brier_val: float) -> str:
                    score = 0
                    if top1_hit >= 0.55: score += 3
                    elif top1_hit >= 0.48: score += 2
                    elif top1_hit >= 0.40: score += 1
                    if not np.isnan(brier_val):
                        if brier_val <= 0.085: score += 3
                        elif brier_val <= 0.100: score += 2
                        elif brier_val <= 0.115: score += 1
                    bands = ["C","B-","B","B+","A-","A","A+"]
                    return bands[min(score, len(bands)-1)]

                grade = letter_grade(top1, brier)

                c1, c2, c3 = st.columns(3)
                c1.metric("Top-1 hit", f"{top1*100:.0f}%")
                c2.metric("Brier (↓ better)", "—" if np.isnan(brier) else f"{brier:.3f}")
                c3.metric("Overall grade", grade)

                show_details = st.toggle("Show per-season details", value=False, key="modelq_details")
                if show_details and per_season is not None and not per_season.empty:
                    from streamlit import column_config as _cc
                    st.dataframe(
                        per_season.assign(
                            log_loss=lambda d: d["log_loss"].round(4),
                            brier=lambda d: d["brier"].round(4),
                            top1_hit=lambda d: (d["top1_hit"]*100).round(1),
                        ),
                        use_container_width=True, hide_index=True,
                        column_config={
                            "season": _cc.NumberColumn("Season", format="%d"),
                            "n_rows": _cc.NumberColumn("# rows", format="%d"),
                            "n_races": _cc.NumberColumn("# races", format="%d"),
                            "log_loss": _cc.NumberColumn("Log loss", format="%.4f"),
                            "brier": _cc.NumberColumn("Brier", format="%.4f"),
                            "top1_hit": _cc.ProgressColumn("Top-1 hit", format="%.1f%%",
                                                        min_value=0.0, max_value=100.0),
                        }
                    )

                st.caption(
                    "Top-1 hit = how often our #1 pick actually won. "
                    "Brier = probability error (lower is better). "
                    "Grade combines both so you don’t have to parse the numbers."
                )
            else:
                st.caption("Click **Check now** to run the backtest. Results will stay visible so you can toggle details.")

st.divider()
with st.expander("ℹ About this model"):
    st.write("This model is trained on historical finishes. Accuracy improves with extra features like driver form, team form, track type, and weather.")

    