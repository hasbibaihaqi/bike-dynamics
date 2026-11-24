from typing import Any, Dict, List
import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Exception khusus untuk kesalahan validasi input pengguna."""


def enrich_and_validate_demand_features(
    raw_input: Dict[str, Any],
    feature_order: List[str],
    defaults: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Memvalidasi input fitur permintaan dan menurunkan fitur rekayasa temporal/siklikal
    secara deterministik agar 100% selaras dengan 02_preprocessing.ipynb.
    """
    defaults = defaults or {}
    data = {**defaults, **(raw_input or {})}

    try:
        season = int(data.get("season", 2))
        weather = int(data.get("weather", 1))
        year = int(data.get("year", 2012))
        month = int(data.get("month", 6))
        day = int(data.get("day", 15))
        hour = int(data.get("hour", 8))
        dayofweek = int(data.get("dayofweek", 2))
        holiday = int(data.get("holiday", 0))
        workingday = int(data.get("workingday", 1))
        temp = float(data.get("temp", 20.0))
        atemp = float(data.get("atemp", temp))
        humidity = float(data.get("humidity", 60.0))
        windspeed = float(data.get("windspeed", 12.0))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Format tipe data input tidak valid: {exc}") from exc

    if season not in {1, 2, 3, 4}:
        raise ValidationError("Nilai 'season' harus berada pada rentang 1 s.d. 4.")
    if weather not in {1, 2, 3, 4}:
        raise ValidationError("Nilai 'weather' harus berada pada rentang 1 s.d. 4.")
    if not (0 <= hour <= 23):
        raise ValidationError("Nilai 'hour' harus berada pada rentang 0 s.d. 23.")
    if not (1 <= month <= 12):
        raise ValidationError("Nilai 'month' harus berada pada rentang 1 s.d. 12.")
    if not (1 <= day <= 31):
        raise ValidationError("Nilai 'day' harus berada pada rentang 1 s.d. 31.")
    if not (0 <= dayofweek <= 6):
        raise ValidationError("Nilai 'dayofweek' harus berada pada rentang 0 s.d. 6.")
    if holiday not in {0, 1} or workingday not in {0, 1}:
        raise ValidationError("Nilai 'holiday' dan 'workingday' harus 0 atau 1.")
    if not (-20.0 <= temp <= 55.0):
        raise ValidationError("Nilai 'temp' di luar batas wajar (-20°C s.d. 55°C).")
    if not (0.0 <= humidity <= 100.0):
        raise ValidationError("Nilai 'humidity' harus berada pada rentang 0% s.d. 100%.")
    if windspeed < 0.0:
        raise ValidationError("Nilai 'windspeed' tidak boleh negatif.")

    # Penurunan fitur rekayasa (konsisten dengan Tahap 2)
    is_weekend = int(data.get("is_weekend", 1 if dayofweek >= 5 else 0))
    workday_peak = (workingday == 1) and (hour in {7, 8, 9, 17, 18, 19})
    leisure_peak = (workingday == 0) and (11 <= hour <= 16)
    is_rush_hour = int(data.get("is_rush_hour", 1 if (workday_peak or leisure_peak) else 0))

    hour_sin = float(np.sin(2.0 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2.0 * np.pi * hour / 24.0))
    month_sin = float(np.sin(2.0 * np.pi * month / 12.0))
    month_cos = float(np.cos(2.0 * np.pi * month / 12.0))
    is_windspeed_zero = int(1 if windspeed == 0.0 else 0)

    full_row = {
        "season": season,
        "weather": weather,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "dayofweek": dayofweek,
        "holiday": holiday,
        "workingday": workingday,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "temp": temp,
        "atemp": atemp,
        "humidity": humidity,
        "windspeed": windspeed,
        "is_windspeed_zero": is_windspeed_zero,
    }

    missing_cols = [col for col in feature_order if col not in full_row]
    if missing_cols:
        raise ValidationError(f"Fitur wajib tidak lengkap: {missing_cols}")

    return pd.DataFrame([{col: full_row[col] for col in feature_order}])


def validate_simulation_params(params: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Memvalidasi parameter input untuk simulasi System Dynamics dan menjamin
    keseimbangan stok awal (initial_available + initial_rented + initial_maintenance == total_fleet).
    """
    merged = {**defaults, **(params or {})}
    try:
        total_fleet = float(merged["total_fleet"])
        initial_available = float(merged["initial_available"])
        initial_rented = float(merged.get("initial_rented", 0.0))
        initial_maintenance = float(merged["initial_maintenance"])
        rental_duration_hours = float(merged["rental_duration_hours"])
        damage_rate = float(merged["damage_rate"])
        repair_duration_hours = float(merged["repair_duration_hours"])
        demand_multiplier = float(merged["demand_multiplier"])
        simulation_hours = int(merged["simulation_hours"])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValidationError(f"Parameter simulasi tidak valid: {exc}") from exc

    if total_fleet <= 0:
        raise ValidationError("Total armada (total_fleet) harus lebih besar dari 0.")
    if initial_available < 0 or initial_rented < 0 or initial_maintenance < 0:
        raise ValidationError("Nilai stok awal sepeda tidak boleh negatif.")
    if initial_available + initial_maintenance > total_fleet + 1e-6:
        raise ValidationError(
            "Jumlah sepeda tersedia awal + sepeda perawatan awal tidak boleh melebihi total armada."
        )

    # Jika initial_rented tidak diisi eksplisit oleh form, hitung otomatis dari sisa total_fleet
    if "initial_rented" not in (params or {}) or params.get("initial_rented") in (None, ""):
        initial_rented = max(0.0, total_fleet - initial_available - initial_maintenance)
    elif abs((initial_available + initial_rented + initial_maintenance) - total_fleet) > 1e-3:
        # Sesuaikan proporsional agar hukum konservasi armada tetap terpenuhi
        initial_rented = max(0.0, total_fleet - initial_available - initial_maintenance)

    if not (0.25 <= rental_duration_hours <= 48.0):
        raise ValidationError("Durasi sewa harus antara 0.25 jam hingga 48 jam.")
    if not (0.0 <= damage_rate <= 1.0):
        raise ValidationError("Tingkat kerusakan (damage_rate) harus antara 0.0 dan 1.0 (0% - 100%).")
    if not (0.5 <= repair_duration_hours <= 168.0):
        raise ValidationError("Durasi perbaikan harus antara 0.5 jam hingga 168 jam.")
    if not (0.1 <= demand_multiplier <= 5.0):
        raise ValidationError("Pengali permintaan (demand_multiplier) harus antara 0.1 hingga 5.0.")
    if not (1 <= simulation_hours <= 720):
        raise ValidationError("Periode simulasi harus antara 1 hingga 720 jam.")

    return {
        "total_fleet": round(total_fleet, 2),
        "initial_available": round(initial_available, 2),
        "initial_rented": round(initial_rented, 2),
        "initial_maintenance": round(initial_maintenance, 2),
        "rental_duration_hours": round(rental_duration_hours, 2),
        "damage_rate": round(damage_rate, 4),
        "repair_duration_hours": round(repair_duration_hours, 2),
        "demand_multiplier": round(demand_multiplier, 4),
        "simulation_hours": simulation_hours,
    }
