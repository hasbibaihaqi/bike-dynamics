from pathlib import Path
from flask import current_app, jsonify, render_template, request


def wants_json_response() -> bool:
    if request.args.get("format") == "json":
        return True
    best = request.accept_mimetypes.best_match(["application/json", "text/html"])
    return best == "application/json" and (
        request.accept_mimetypes[best] > request.accept_mimetypes["text/html"]
    )


def render_or_fallback(template_name: str, page_title: str, payload: dict):
    """
    Merender template Jinja2 jika file HTML di templates/ sudah tersedia (Tahap 8).
    Jika belum ada (saat pengujian backend Tahap 7) atau client meminta JSON,
    mengembalikan JSON / tampilan diagnostik backend yang bersih tanpa error.
    """
    if wants_json_response():
        return jsonify(payload)

    templates_dir = Path(current_app.config["TEMPLATES_DIR"])
    if (templates_dir / template_name).exists():
        return render_template(template_name, page_title=page_title, **payload)

    # Fallback halaman status backend saat Tahap 7 sebelum Tahap 8 (templates/) dibuat
    nav_links = [
        ("/", "1. Dashboard"),
        ("/dataset", "2. Dataset Management"),
        ("/demand-analysis", "3. Demand Analysis"),
        ("/model-performance", "4. Model Performance"),
        ("/system-dynamics", "5. System Dynamics"),
        ("/simulation-center", "6. Simulation Center"),
        ("/scenario-analysis", "7. Scenario Analysis"),
        ("/sensitivity-analysis", "8. Sensitivity Analysis"),
        ("/results", "9. Results & Reports"),
    ]
    nav_html = " | ".join(f'<a href="{url}">{label}</a>' for url, label in nav_links)
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{page_title} - BIKE-DYNAMICS</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:28px;background:#f8fafc;color:#0f172a;}"
        ".card{background:#fff;border:1px solid #cbd5e1;border-radius:8px;padding:18px;margin-top:16px;}"
        "a{color:#0284c7;text-decoration:none;font-weight:600;} pre{background:#0f172a;color:#e2e8f0;padding:14px;border-radius:6px;overflow:auto;max-height:420px;}</style>"
        f"</head><body><h2>BIKE-DYNAMICS — Backend Active ({page_title})</h2>"
        f"<div class='card'>{nav_html}</div>"
        f"<div class='card'><p>Backend Flask berjalan normal. Tambahkan <code>?format=json</code> pada URL untuk output JSON murni, atau lanjutkan ke <b>Tahap 8</b> untuk antarmuka Bootstrap 5 + Chart.js.</p>"
        f"<p><strong>Keys tersedia:</strong> {list(payload.keys())}</p></div></body></html>"
    )
