import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import requests
from streamlit import column_config
import streamlit.components.v1 as components
import math


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
    "Great Britain": "🇬🇧", "United Kingdom": "🇬🇧", "Great Britain": "🇬🇧",
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

FETCH_LABEL = "Fetch latest qualifying (OpenF1)"
OPENF1_BASE = "https://api.openf1.org/v1"



st.set_page_config(page_title="F1 Winner Predictor", page_icon="🏁", layout="centered")

# ---- Cache helpers  ----
@st.cache_data
def load_csv(path):
    """Cached CSV reader."""
    return pd.read_csv(path)

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
        try:
            d_resp.raise_for_status()
            djson = d_resp.json()
            for d in djson:
                num = str(_pick_value(d, "driver_number"))
                if not num or num == "None":
                    continue
                drivers[num] = _best_driver_name(d)
                three_letter[num] = _pick_value(d, "three_letter_name")
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

            rows.append({"driver": name, "code": str(code), "grid_quali": qpos, "driver_number": num})

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
        return out[["driver","code","grid_quali","grid"]]

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
        try:
            djson = requests.get(f"{OPENF1_BASE}/drivers", params={"session_key": quali_key}, timeout=25).json()
            for d in djson:
                num = str(_pick_value(d, "driver_number"))
                if not num or num == "None":
                    continue
                drivers[num] = _best_driver_name(d)
                three_letter[num] = _pick_value(d, "three_letter_name")
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
            rows.append({"driver": name, "code": str(code), "grid_quali": pos, "driver_number": num})

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
        return out[["driver","code","grid_quali","grid"]]

    except Exception as e:
        st.error(f"OpenF1 latest fetch failed: {e}")
        return pd.DataFrame(columns=["driver","code","grid_quali","grid"])

SEASON_CIRCUIT_IDX = build_season_circuit_index()

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

</style>
""", unsafe_allow_html=True)



st.title("🏁 F1 Next-Race Winner Predictor")

DATA_PATH = Path("data/processed/baseline_with_features.csv")
if not DATA_PATH.exists():
    st.error("Missing data/processed/baseline_with_features.csv. Run the notebook section that saves df_feat.")
    st.stop()

df = load_csv(DATA_PATH)

# --- inject season/round from data/raw/races.csv if missing ---
# --- inject season/round from data/raw/races.csv if missing (conflict-safe) ---
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

with st.expander("Debug info"):
    st.caption(f"Loaded: {DATA_PATH.resolve()}")
    st.caption(f"wx cols found: {sorted([c for c in df.columns if c.startswith('wx_')])}")
    st.caption(f"has season/round? { {'season','round'}.issubset(df.columns) }")
    if has_wx:
        cov = df[[c for c in wx_cols if c in df.columns]].notna().mean().round(3)
        st.caption(f"wx coverage (fraction non-null): {cov.to_dict()}")



# --- Weather viewer ---
if has_wx and {"season","round"}.issubset(df.columns) and (SEASON_CIRCUIT_IDX):
    if ("season_sel" in locals()) and ("round_num_selected" in locals()) and (season_sel is not None) and (round_num_selected is not None):
        df_wx_candidates = df.dropna(subset=[c for c in wx_cols if c in df.columns], how="all")
        race_rows = df_wx_candidates[
            (df_wx_candidates["season"].astype(str) == str(season_sel)) &
            (df_wx_candidates["round"].astype(str)  == str(round_num_selected))
        ]
        if not race_rows.empty:
            r0 = race_rows.iloc[0]
            st.sidebar.markdown("### Weather (from enriched file)")
            c1, c2 = st.sidebar.columns(2)
            def fmt(v, unit=""):
                return "—" if pd.isna(v) else f"{float(v):.1f}{unit}"
            c1.metric("Temp (mean)", fmt(r0.get("wx_temp_mean_c"), "°C"))
            c2.metric("Temp (max)",  fmt(r0.get("wx_temp_max_c"),  "°C"))
            c1.metric("Precip",      fmt(r0.get("wx_precip_mm"),   " mm"))
            c2.metric("Wind (max)",  fmt(r0.get("wx_windspeed_max_kmh"), " km/h"))
            wet = int(r0.get("wx_is_wet", 0)) if not pd.isna(r0.get("wx_is_wet", np.nan)) else 0
            hot = int(r0.get("wx_is_hot", 0)) if not pd.isna(r0.get("wx_is_hot", np.nan)) else 0
            st.sidebar.caption(f"**Wet?** {'Yes' if wet else 'No'} · **Hot (≥30°C)?** {'Yes' if hot else 'No'}")
        else:
            st.sidebar.caption("_No weather found for this race in the enriched file._")


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
feature_cols = [c for c in num_cols if c in df.columns]
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
        src = st.radio(
            "Grid data source",
            ["Manual table", "Load from local Kaggle CSV", "Upload qualifying CSV", FETCH_LABEL],
            horizontal=False
        )
        # Only show the fetch button/toggle when the OpenF1 option is selected
        if src == FETCH_LABEL:
            fetch_click = st.button("Fetch now", key="fetch_quali_btn")
            use_latest = st.toggle("Use latest meeting (ignore Season/Round)", value=False, key="openf1_latest")
        else:
            fetch_click = False
            use_latest = False



if SEASON_CIRCUIT_IDX and (season_sel is not None) and (round_num_selected is not None):
    t = p = np.nan
    w = 0
    country = None

    if has_wx and {"season","round"}.issubset(df.columns):
        row = df[
            (df["season"].astype(str) == str(season_sel)) &
            (df["round"].astype(str) == str(round_num_selected))
        ]
        if not row.empty:
            r0 = row.iloc[-1]
            t = r0.get("wx_temp_mean_c", np.nan)
            p = r0.get("wx_precip_mm", np.nan)
            w = int(r0.get("wx_is_wet", 0)) if not pd.isna(r0.get("wx_is_wet", np.nan)) else 0
            # Try to get country info if present in races data
            country = r0.get("country", None)
            
    if CIRCUITS_DF is not None and circuit_id_selected is not None and "country" in CIRCUITS_DF.columns:
        match = CIRCUITS_DF.loc[CIRCUITS_DF["circuitId"] == circuit_id_selected, "country"]
        if not match.empty:
            country = str(match.iloc[0])


    # Get flag emoji
    flag = CIRCUIT_FLAGS.get(str(country), "") if country else ""

    chips = "".join([
        f'<span class="wx-pill">Temp {t:.1f}°C</span>' if not pd.isna(t) else "",
        f'<span class="wx-pill">Rain {p:.1f} mm</span>' if not pd.isna(p) else "",
        f'<span class="wx-pill">{"Wet" if w else "Dry"}</span>',
    ])

    header_html = f"""
    <div class="banner">
      <h3>{flag} Season {season_sel} — {lab}</h3>
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
        st.info("Set inputs on the left and click **Predict**.")
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
    st.subheader("Full Grid: Predict Winner Among Multiple Drivers")

    # --- Session init for persistence across reruns ---
    if "grid_df" not in st.session_state:
        st.session_state.grid_df = None
    if "grid_src" not in st.session_state:
        st.session_state.grid_src = None
    if "last_ctx" not in st.session_state:
        st.session_state.last_ctx = (season_sel, round_num_selected)

    # Reset auto-loaded grids when season/round changes
    if (season_sel, round_num_selected) != st.session_state.last_ctx:
        if st.session_state.grid_src in ("kaggle", "ergast"):
            st.session_state.grid_df = None
            st.session_state.grid_src = None
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
        if SEASON_CIRCUIT_IDX and season_sel is not None and round_num_selected is not None:
            st.write(f"Using **Season {season_sel}** – {lab}")
            if st.button("Load from local CSV"):
                try:
                    auto_df = load_local_quali(int(season_sel), int(round_num_selected))
                    if auto_df.empty:
                        st.warning("No qualifying rows found for that season/circuit. Check your Kaggle coverage.")
                    else:
                        st.success(f"Loaded {len(auto_df)} entries from data/raw/*.csv")
                        user_df = auto_df
                        st.session_state.grid_df = auto_df
                        st.session_state.grid_src = "kaggle"

                        # (Optional) show weather snippet from enriched dataset
                        if has_wx and "season" in df.columns and "round" in df.columns:
                            cand = df[
                                (df["season"].astype(str) == str(season_sel)) &
                                (df["round"].astype(str)  == str(round_num_selected))
                            ].dropna(subset=[c for c in wx_cols if c in df.columns], how="all")
                            if not cand.empty:
                                r0 = cand.iloc[0]
                                st.info("Weather from enriched dataset for this race:")
                                colA, colB, colC, colD = st.columns(4)
                                fmt = lambda v,u="": "—" if pd.isna(v) else f"{float(v):.1f}{u}"
                                colA.metric("Temp (mean)", fmt(r0.get("wx_temp_mean_c"), "°C"))
                                colB.metric("Temp (max)",  fmt(r0.get("wx_temp_max_c"),  "°C"))
                                colC.metric("Precip",      fmt(r0.get("wx_precip_mm"),   " mm"))
                                colD.metric("Wind (max)",  fmt(r0.get("wx_windspeed_max_kmh"), " km/h"))
                                wet = int(r0.get("wx_is_wet", 0)) if not pd.isna(r0.get("wx_is_wet", np.nan)) else 0
                                hot = int(r0.get("wx_is_hot", 0)) if not pd.isna(r0.get("wx_is_hot", np.nan)) else 0
                                st.caption(f"**Wet?** {'Yes' if wet else 'No'} · **Hot (≥30°C)?** {'Yes' if hot else 'No'}")
                except Exception as e:
                    st.error(f"Local load failed: {e}")
                    st.info("Ensure data/raw/races.csv, qualifying.csv, and drivers.csv exist (from your Kaggle download).")
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

    # --- Stable key to avoid resets on tweaks (only one editor in app) ---
    editor_key = "grid_editor_fullgrid"

    # Editor (seeded from session), and persist edits back to session
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
    # Persist the edited table so it survives any widget change
    st.session_state.grid_df = edit_df

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
    if "driver" in pred_rows.columns and (BY_CODE or BY_NAME):
        pred_rows["driverId"] = pred_rows["driver"].apply(lambda v: map_driver_to_id_from_text(v) if isinstance(v, str) else None)
    if "driverId" in pred_rows.columns:
        pred_rows["driverId"] = pd.to_numeric(pred_rows["driverId"], errors="coerce").astype("Int64")
    if not driver_feats_ref.empty and "driverId" in driver_feats_ref.columns:
        driver_feats_ref["driverId"] = pd.to_numeric(driver_feats_ref["driverId"], errors="coerce").astype("Int64")
    if "driverId" in pred_rows.columns and not driver_feats_ref.empty:
        pred_rows = pred_rows.merge(driver_feats_ref, on="driverId", how="left")

    if circuit_id_selected is not None:
        pred_rows["circuitId"] = int(circuit_id_selected)
        st.caption(f"Predicting for: **Season {season_sel}** — {lab}")

    if {"driverId","circuitId"}.issubset(pred_rows.columns) and not driver_circuit_feats_ref.empty:
        driver_circuit_feats_ref["driverId"] = pd.to_numeric(driver_circuit_feats_ref["driverId"], errors="coerce").astype("Int64")
        driver_circuit_feats_ref["circuitId"] = pd.to_numeric(driver_circuit_feats_ref["circuitId"], errors="coerce").astype("Int64")
        pred_rows["circuitId"] = pd.to_numeric(pred_rows["circuitId"], errors="coerce").astype("Int64")
        pred_rows = pred_rows.merge(driver_circuit_feats_ref, on=["driverId","circuitId"], how="left")

    # --- BASE matrix & predict ---
    X_base_df = pred_rows.reindex(columns=feature_cols).fillna(feature_defaults).fillna(0.0)
    X_base = np.nan_to_num(X_base_df.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    proba_base = clf.predict_proba(X_base)[:, 1]

    table_df = pred_rows.copy()
    table_df["win_prob"] = proba_base
    table_df = table_df.sort_values("win_prob", ascending=False).reset_index(drop=True)
    table_df["win_pct"] = table_df["win_prob"] * 100.0

        # --- INIT (once) and READ what-if state ---
    if "whatif_delta_grid" not in st.session_state:
        st.session_state["whatif_delta_grid"] = 0
    if "whatif_wet" not in st.session_state:
        # default from race weather if available; else False
        st.session_state["whatif_wet"] = (bool(w) if 'w' in locals() and not pd.isna(w) else False)
    if "whatif_hot" not in st.session_state:
        base_hot = (float(t) >= 30.0) if 't' in locals() and not pd.isna(t) else False
        st.session_state["whatif_hot"] = base_hot

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
    tuned["win_prob"] = clf.predict_proba(X_tuned)[:, 1]
    tuned = tuned.sort_values("win_prob", ascending=False).reset_index(drop=True)
    tuned["win_pct"] = tuned["win_prob"] * 100.0

    table_df = tuned  # <- show tweaked table always

    # --- TABLE FIRST (unchanged) ---
    display_df = table_df.copy()
    show_cols = [c for c in ["driver","grid","grid_quali","wx_is_wet","wx_temp_max_c"] if c in display_df.columns] + ["win_pct"]
    display_cols = [c for c in show_cols if c in display_df.columns]

    st.write("### Predicted Win Probabilities")
    st.dataframe(
        display_df[display_cols],
        use_container_width=True,
        column_config={
            **({"grid": column_config.NumberColumn("Grid", format="%d")} if "grid" in display_cols else {}),
            **({"grid_quali": column_config.NumberColumn("Quali", format="%d")} if "grid_quali" in display_cols else {}),
            "win_pct": column_config.ProgressColumn("Win Prob", format="%.1f%%", min_value=0.0, max_value=100.0),
        },
        hide_index=True,
    )

    # === WHAT-IF CONTROLS (REPLACEMENT) ===
    # Big title you fully control
    st.markdown("""
    <style>
    .whatif-title {
        font-size: 1.5rem;     /* change as you like */
        font-weight: 800;
        letter-spacing: .2px;
        margin: 6px 0 4px;
        line-height: 1.3;
    }
    .biglabel { 
        font-size: 1.08rem; 
        font-weight: 700; 
        margin: 6px 0 4px; 
        line-height: 1.35; 
    }
    .subhead  { 
        font-size: 1.28rem; 
        font-weight: 700; 
        margin: 10px 0 4px; 
    }
    </style>
    <div class="whatif-title">🔧 Tweak inputs</div>
    """, unsafe_allow_html=True)

    # Use an empty-label expander so only the big title shows
    with st.expander("", expanded=True):  # set False if you want it closed by default
        # Grid slider with a custom large label
        st.markdown('<div class="biglabel">Grid adjustment (±)</div>', unsafe_allow_html=True)
        st.slider(
            label="", min_value=-3, max_value=3, step=1,
            value=dgrid, key="whatif_delta_grid", label_visibility="collapsed"
        )

        # Toggles with custom large labels
        st.markdown('<div class="subhead">Race conditions</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="biglabel">Wet race</div>', unsafe_allow_html=True)
            st.toggle("", value=wet_flag, key="whatif_wet", label_visibility="collapsed")
        with c2:
            st.markdown('<div class="biglabel">Hot (≥30°C)</div>', unsafe_allow_html=True)
            st.toggle("", value=hot_flag, key="whatif_hot", label_visibility="collapsed")

        st.caption("Changes update the table above automatically.")

    # --- Winner + download reflects current table ---
    top_row = table_df.iloc[0]
    winner_name = top_row["driver"] if "driver" in table_df.columns and pd.notna(top_row["driver"]) else f"Grid {int(top_row['grid'])}"
    st.success(f"🏆 Predicted winner: **{winner_name}**  (Win prob: {top_row['win_prob'] * 100:.1f}%)")

    st.download_button(
        "⬇️ Download predictions (CSV)",
        table_df[display_cols + (["win_prob"] if "win_prob" not in display_cols else [])].to_csv(index=False).encode(),
        file_name=f"f1_predictions_s{season_sel}_r{round_num_selected}.csv" if (SEASON_CIRCUIT_IDX and season_sel is not None and round_num_selected is not None) else "f1_predictions.csv",
        mime="text/csv",
    )

st.divider()
with st.expander("ℹ About this model"):
    st.write("This model is trained on historical finishes. Accuracy improves with extra features like driver form, team form, track type, and weather.")

    