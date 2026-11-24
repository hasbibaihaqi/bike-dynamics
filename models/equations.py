from typing import Tuple

WEATHER_DAMAGE_MULTIPLIERS = {
    1: 1.00,  # Clear / Partly Cloudy
    2: 1.10,  # Mist / Cloudy
    3: 1.35,  # Light Snow / Rain
    4: 1.60,  # Heavy Rain / Ice
}


def compute_rental_rate(available_bikes: float, demand_rate: float) -> float:
    """R(t) = min(B_a(t), D(t)) — Penyewaan aktual dibatasi oleh ketersediaan sepeda."""
    return max(0.0, min(float(available_bikes), float(demand_rate)))


def compute_unmet_demand(demand_rate: float, rental_rate: float) -> float:
    """D_unmet(t) = max(0, D(t) - R(t)) — Permintaan tidak terpenuhi."""
    return max(0.0, float(demand_rate) - float(rental_rate))


def compute_return_rate(rented_bikes: float, rental_duration_hours: float, dt: float = 1.0) -> float:
    """U(t) = B_r(t) * min(1.0, dt / tau_r) — Laju pengembalian sepeda per langkah waktu."""
    fraction = min(1.0, float(dt) / max(float(rental_duration_hours), 0.1))
    return max(0.0, float(rented_bikes) * fraction)


def compute_effective_damage_rate(base_damage_rate: float, weather_condition: int) -> float:
    """alpha_d_eff(t) = min(0.95, alpha_d * phi_w(W(t))) — Pengaruh cuaca terhadap kerusakan."""
    w_mult = WEATHER_DAMAGE_MULTIPLIERS.get(int(weather_condition), 1.0)
    return min(0.95, max(0.0, float(base_damage_rate)) * w_mult)


def compute_maintenance_inflow(return_rate: float, effective_damage_rate: float) -> float:
    """M_i(t) = U(t) * alpha_d_eff(t) — Laju sepeda masuk perawatan."""
    return max(0.0, float(return_rate) * float(effective_damage_rate))


def compute_maintenance_outflow(maintenance_bikes: float, repair_duration_hours: float, dt: float = 1.0) -> float:
    """M_o(t) = B_m(t) * min(1.0, dt / tau_m) — Laju sepeda selesai diperbaiki."""
    fraction = min(1.0, float(dt) / max(float(repair_duration_hours), 0.1))
    return max(0.0, float(maintenance_bikes) * fraction)


def step_stocks(
    available_bikes: float,
    rented_bikes: float,
    maintenance_bikes: float,
    rental_rate: float,
    return_rate: float,
    maint_inflow: float,
    maint_outflow: float,
) -> Tuple[float, float, float]:
    """
    Persamaan diferensi Stock and Flow:
    B_a(t+1) = B_a(t) - R(t) + U(t) - M_i(t) + M_o(t)
    B_r(t+1) = B_r(t) + R(t) - U(t)
    B_m(t+1) = B_m(t) + M_i(t) - M_o(t)
    """
    ba_next = available_bikes - rental_rate + return_rate - maint_inflow + maint_outflow
    br_next = rented_bikes + rental_rate - return_rate
    bm_next = maintenance_bikes + maint_inflow - maint_outflow

    # Penjagaan presisi floating-point agar tidak bernilai -0.0
    ba_next = max(0.0, ba_next)
    br_next = max(0.0, br_next)
    bm_next = max(0.0, bm_next)
    return ba_next, br_next, bm_next
