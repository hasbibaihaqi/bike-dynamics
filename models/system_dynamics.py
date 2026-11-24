from typing import Any, Dict, List, Sequence
import numpy as np

from models.equations import (
    WEATHER_DAMAGE_MULTIPLIERS,
    compute_effective_damage_rate,
    compute_maintenance_inflow,
    compute_maintenance_outflow,
    compute_rental_rate,
    compute_return_rate,
    compute_unmet_demand,
    step_stocks,
)


class SystemDynamicsSimulator:
    """
    Mesin simulasi System Dynamics (Stock and Flow) untuk pengelolaan armada Bike Sharing.
    """

    STOCK_VARIABLES = [
        {
            "name": "Available Bikes",
            "symbol": "B_a(t)",
            "unit": "Sepeda",
            "description": "Jumlah sepeda siap sewa yang tersedia di stasiun pada waktu t.",
        },
        {
            "name": "Rented Bikes",
            "symbol": "B_r(t)",
            "unit": "Sepeda",
            "description": "Jumlah sepeda yang sedang aktif digunakan oleh pelanggan pada waktu t.",
        },
        {
            "name": "Maintenance Bikes",
            "symbol": "B_m(t)",
            "unit": "Sepeda",
            "description": "Jumlah sepeda yang sedang dalam proses perawatan atau perbaikan.",
        },
    ]

    FLOW_VARIABLES = [
        {
            "name": "Rental Rate",
            "symbol": "R(t)",
            "unit": "Sepeda/jam",
            "description": "Laju penyewaan aktual yang berpindah dari Available Bikes ke Rented Bikes.",
        },
        {
            "name": "Return Rate",
            "symbol": "U(t)",
            "unit": "Sepeda/jam",
            "description": "Laju pengembalian sepeda selesai sewa dari Rented Bikes.",
        },
        {
            "name": "Maintenance Inflow",
            "symbol": "M_i(t)",
            "unit": "Sepeda/jam",
            "description": "Laju sepeda yang mengalami kerusakan saat pengembalian menuju Maintenance Bikes.",
        },
        {
            "name": "Maintenance Outflow",
            "symbol": "M_o(t)",
            "unit": "Sepeda/jam",
            "description": "Laju sepeda selesai diperbaiki dari Maintenance Bikes kembali ke Available Bikes.",
        },
    ]

    AUXILIARY_VARIABLES = [
        {"name": "Demand Rate", "symbol": "D(t)", "unit": "Sepeda/jam", "description": "Permintaan potensial dari prediksi model ML."},
        {"name": "Fleet Capacity", "symbol": "B_total", "unit": "Sepeda", "description": "Total kapasitas armada (B_a + B_r + B_m)."},
        {"name": "Rental Duration", "symbol": "tau_r", "unit": "Jam", "description": "Rata-rata lama durasi penyewaan sepeda."},
        {"name": "Damage Rate", "symbol": "alpha_d", "unit": "Fraksi", "description": "Proporsi sepeda dikembalikan yang butuh perawatan."},
        {"name": "Repair Duration", "symbol": "tau_m", "unit": "Jam", "description": "Rata-rata durasi perbaikan unit di bengkel."},
        {"name": "Weather Condition", "symbol": "W(t)", "unit": "Kategori (1-4)", "description": "Kondisi cuaca yang memengaruhi permintaan dan risiko kerusakan."},
        {"name": "Fleet Utilization", "symbol": "U_util(t)", "unit": "Persen (%)", "description": "Persentase armada yang sedang disewa terhadap kapasitas total."},
        {"name": "Unmet Demand", "symbol": "D_unmet(t)", "unit": "Sepeda/jam", "description": "Permintaan tidak terpenuhi akibat stok Available Bikes habis."},
    ]

    EQUATIONS_SPEC = [
        {"title": "Available Bikes Stock", "latex": "B_a(t+1) = B_a(t) - R(t) + U(t) - M_i(t) + M_o(t)"},
        {"title": "Rented Bikes Stock", "latex": "B_r(t+1) = B_r(t) + R(t) - U(t)"},
        {"title": "Maintenance Bikes Stock", "latex": "B_m(t+1) = B_m(t) + M_i(t) - M_o(t)"},
        {"title": "Konservasi Total Armada", "latex": "B_total = B_a(t) + B_r(t) + B_m(t)"},
        {"title": "Rental Rate & Unmet Demand", "latex": "R(t) = min(B_a(t), D(t)),   D_unmet(t) = max(0, D(t) - R(t))"},
        {"title": "Return Rate", "latex": "U(t) = B_r(t) * min(1, dt / tau_r)"},
        {"title": "Maintenance Inflow & Outflow", "latex": "M_i(t) = U(t) * min(0.95, alpha_d * phi_w(W(t))),   M_o(t) = B_m(t) * min(1, dt / tau_m)"},
    ]

    def simulate(
        self,
        demand_series: Sequence[float],
        weather_series: Sequence[int] | None = None,
        timestamps: Sequence[str] | None = None,
        total_fleet: float = 600.0,
        initial_available: float = 510.0,
        initial_rented: float = 60.0,
        initial_maintenance: float = 30.0,
        rental_duration_hours: float = 1.5,
        damage_rate: float = 0.04,
        repair_duration_hours: float = 6.0,
        demand_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        demand_arr = np.asarray(demand_series, dtype=float) * float(demand_multiplier)
        n_steps = len(demand_arr)
        if n_steps == 0:
            raise ValueError("Deret permintaan (demand_series) tidak boleh kosong.")

        if weather_series is None:
            weather_arr = np.ones(n_steps, dtype=int)
        else:
            weather_arr = np.asarray(weather_series, dtype=int)

        # Penyelarasan stok awal agar total konservasi konsisten
        init_sum = float(initial_available + initial_rented + initial_maintenance)
        if abs(init_sum - total_fleet) > 1e-6:
            scale = float(total_fleet) / init_sum if init_sum > 0 else 1.0
            initial_available *= scale
            initial_rented *= scale
            initial_maintenance *= scale

        ba = float(initial_available)
        br = float(initial_rented)
        bm = float(initial_maintenance)
        dt = 1.0

        timeseries: List[Dict[str, Any]] = []
        max_conservation_error = 0.0

        for t in range(n_steps):
            pot_demand = max(0.0, float(demand_arr[t]))
            w_cond = int(weather_arr[t])

            rental_rate = compute_rental_rate(ba, pot_demand)
            unmet_demand = compute_unmet_demand(pot_demand, rental_rate)
            return_rate = compute_return_rate(br, rental_duration_hours, dt)
            eff_damage = compute_effective_damage_rate(damage_rate, w_cond)
            maint_inflow = compute_maintenance_inflow(return_rate, eff_damage)
            maint_outflow = compute_maintenance_outflow(bm, repair_duration_hours, dt)

            ba_next, br_next, bm_next = step_stocks(
                ba, br, bm, rental_rate, return_rate, maint_inflow, maint_outflow
            )
            total_next = ba_next + br_next + bm_next
            max_conservation_error = max(max_conservation_error, abs(total_next - total_fleet))
            util_pct = (br_next / total_fleet * 100.0) if total_fleet > 0 else 0.0

            timeseries.append(
                {
                    "step": t,
                    "datetime": str(timestamps[t]) if timestamps is not None else f"Jam ke-{t+1}",
                    "weather": w_cond,
                    "demand_rate": round(pot_demand, 2),
                    "rental_rate": round(rental_rate, 2),
                    "unmet_demand": round(unmet_demand, 2),
                    "return_rate": round(return_rate, 2),
                    "maint_inflow": round(maint_inflow, 2),
                    "maint_outflow": round(maint_outflow, 2),
                    "available_bikes": round(ba_next, 2),
                    "rented_bikes": round(br_next, 2),
                    "maintenance_bikes": round(bm_next, 2),
                    "total_fleet": round(total_next, 4),
                    "utilization_pct": round(util_pct, 2),
                }
            )
            ba, br, bm = ba_next, br_next, bm_next

        total_demand = sum(r["demand_rate"] for r in timeseries)
        total_rentals = sum(r["rental_rate"] for r in timeseries)
        total_unmet = sum(r["unmet_demand"] for r in timeseries)
        service_level = (total_rentals / total_demand * 100.0) if total_demand > 0 else 100.0

        avg_available = float(np.mean([r["available_bikes"] for r in timeseries]))
        avg_rented = float(np.mean([r["rented_bikes"] for r in timeseries]))
        avg_maintenance = float(np.mean([r["maintenance_bikes"] for r in timeseries]))
        avg_utilization = float(np.mean([r["utilization_pct"] for r in timeseries]))
        stockout_hours = int(sum(1 for r in timeseries if r["available_bikes"] < 1.0))

        last_step = timeseries[-1]

        return {
            "summary": {
                "simulation_hours": n_steps,
                "total_fleet": round(float(total_fleet), 2),
                "final_available_bikes": last_step["available_bikes"],
                "final_rented_bikes": last_step["rented_bikes"],
                "final_maintenance_bikes": last_step["maintenance_bikes"],
                "total_demand": round(total_demand, 2),
                "total_rentals": round(total_rentals, 2),
                "total_unmet_demand": round(total_unmet, 2),
                "service_level_pct": round(service_level, 2),
                "avg_available_bikes": round(avg_available, 2),
                "avg_rented_bikes": round(avg_rented, 2),
                "avg_maintenance_bikes": round(avg_maintenance, 2),
                "avg_utilization_pct": round(avg_utilization, 2),
                "stockout_hours": stockout_hours,
                "conservation_verified": bool(max_conservation_error < 1e-5),
                "max_conservation_error": round(max_conservation_error, 8),
            },
            "timeseries": timeseries,
        }

    @classmethod
    def get_model_documentation(cls, assumptions: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "stocks": cls.STOCK_VARIABLES,
            "flows": cls.FLOW_VARIABLES,
            "auxiliaries": cls.AUXILIARY_VARIABLES,
            "equations": cls.EQUATIONS_SPEC,
            "weather_multipliers": WEATHER_DAMAGE_MULTIPLIERS,
            "assumptions": assumptions,
        }
