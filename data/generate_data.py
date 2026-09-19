"""
Generate a realistic synthetic dataset for supply chain shipment delay prediction.

Since no live shipment-level dataset is bundled here, this creates a
statistically realistic dataset with known ground-truth relationships
(so the models below have real signal to learn, not noise).
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 15000

carriers = ["MaerskLine", "MSC", "CMA_CGM", "COSCO", "Hapag_Lloyd", "ONE_Line", "Evergreen"]
carrier_reliability = {  # base reliability score (higher = more reliable)
    "MaerskLine": 0.85, "MSC": 0.78, "CMA_CGM": 0.75, "COSCO": 0.68,
    "Hapag_Lloyd": 0.82, "ONE_Line": 0.72, "Evergreen": 0.70,
}

modes = ["Sea", "Air", "Rail", "Road"]
mode_base_delay_rate = {"Sea": 0.28, "Air": 0.09, "Rail": 0.16, "Road": 0.14}

origin_ports = ["Shanghai", "Singapore", "Rotterdam", "Los_Angeles", "Busan",
                "Hamburg", "Dubai", "Antwerp", "Ningbo", "Hong_Kong"]
dest_ports = ["New_York", "Long_Beach", "Felixstowe", "Santos", "Mumbai",
              "Sydney", "Tokyo", "Vancouver", "Le_Havre", "Durban"]

port_congestion_index = {p: np.random.uniform(0.1, 0.9) for p in origin_ports + dest_ports}

seasons = ["Q1", "Q2", "Q3", "Q4"]
season_congestion_multiplier = {"Q1": 1.0, "Q2": 0.9, "Q3": 1.15, "Q4": 1.35}  # peak-season effect

rows = []
for i in range(N):
    carrier = np.random.choice(carriers)
    mode = np.random.choice(modes, p=[0.55, 0.15, 0.10, 0.20])
    origin = np.random.choice(origin_ports)
    dest = np.random.choice(dest_ports)
    season = np.random.choice(seasons)

    distance_km = np.random.uniform(500, 20000) if mode in ("Sea", "Air") else np.random.uniform(100, 3000)
    planned_transit_days = {
        "Sea": distance_km / 550, "Air": distance_km / 4500,
        "Rail": distance_km / 500, "Road": distance_km / 550,
    }[mode] + np.random.uniform(1, 4)

    origin_congestion = port_congestion_index[origin] * season_congestion_multiplier[season]
    dest_congestion = port_congestion_index[dest] * season_congestion_multiplier[season]

    customs_complexity = np.random.uniform(0, 1)  # 0 = simple paperwork, 1 = high scrutiny
    num_customs_stops = np.random.poisson(1.2) if mode in ("Sea", "Road") else np.random.poisson(0.4)
    weather_risk = np.random.uniform(0, 1)  # storm/monsoon season proxy
    reliability = carrier_reliability[carrier]
    shipment_value_usd = np.exp(np.random.uniform(7, 13))  # $1k - $440k, log-normal-ish
    is_hazmat = np.random.choice([0, 1], p=[0.92, 0.08])
    num_prior_shipments_route = np.random.poisson(8)

    # ---- ground truth delay-risk model (logit) ----
    logit = (
        -3.3
        + 2.2 * mode_base_delay_rate[mode]
        + 1.4 * origin_congestion
        + 1.3 * dest_congestion
        + 1.1 * customs_complexity
        + 0.35 * num_customs_stops
        + 0.9 * weather_risk
        - 2.0 * (reliability - 0.5)
        + 0.6 * is_hazmat
        - 0.15 * np.log1p(num_prior_shipments_route)
        + np.random.normal(0, 0.5)  # irreducible noise
    )
    prob_delay = 1 / (1 + np.exp(-logit))
    is_delayed = np.random.binomial(1, prob_delay)

    # delay magnitude (days) - only meaningful when delayed, but we generate for all
    delay_days = 0.0
    if is_delayed:
        delay_days = max(0.5, np.random.gamma(
            shape=2.0,
            scale=1.5 + 3 * origin_congestion + 3 * dest_congestion + 2 * customs_complexity + 2 * weather_risk
        ))

    rows.append({
        "shipment_id": f"SHP{100000+i}",
        "carrier": carrier,
        "mode": mode,
        "origin_port": origin,
        "dest_port": dest,
        "season": season,
        "distance_km": round(distance_km, 1),
        "planned_transit_days": round(planned_transit_days, 1),
        "origin_congestion_idx": round(origin_congestion, 3),
        "dest_congestion_idx": round(dest_congestion, 3),
        "customs_complexity": round(customs_complexity, 3),
        "num_customs_stops": num_customs_stops,
        "weather_risk": round(weather_risk, 3),
        "carrier_reliability_score": round(reliability, 2),
        "shipment_value_usd": round(shipment_value_usd, 2),
        "is_hazmat": is_hazmat,
        "num_prior_shipments_route": num_prior_shipments_route,
        "is_delayed": is_delayed,
        "delay_days": round(delay_days, 2),
    })

df = pd.DataFrame(rows)
df.to_csv("/home/claude/supply_chain_delay_predictor/data/shipments.csv", index=False)
print(df.shape)
print(df["is_delayed"].value_counts(normalize=True))
print(df.head())
