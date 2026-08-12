"""Prediction-versus-actual map payloads, independent of Streamlit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..dataset import SplitManifest, parse_trajectory, pseudonymize_identifier
from ..geo import GridEncoder, haversine_km


_ASSET_DIR = Path(__file__).resolve().parent / "assets"

_OVERVIEW_SPLITS = {
    "train": {"label": "Training", "color": "#0B63CE"},
    "validation": {"label": "Validation", "color": "#D97706"},
    "test": {"label": "Final test", "color": "#C0264A"},
}


def build_trip_overview_payload(
    trips: pd.DataFrame,
    manifest: SplitManifest,
) -> dict[str, Any]:
    """Build a map-ready representation of every accepted model trip."""
    required = {"VIN", "trip_id", "Trajectory"}
    missing = sorted(required - set(trips.columns))
    if missing:
        raise ValueError(f"Trips table is missing required columns: {missing}")
    if trips.empty:
        raise ValueError("Cannot build the overview map from an empty trips table")

    routes: list[dict[str, Any]] = []
    split_counts = {name: 0 for name in _OVERVIEW_SPLITS}
    split_vins = {name: set() for name in _OVERVIEW_SPLITS}
    bounds: list[list[float]] = []

    for _, row in trips.iterrows():
        vin = str(row["VIN"])
        split = manifest.split_for(vin)
        trajectory = parse_trajectory(row["Trajectory"])
        path = [[float(point[1]), float(point[0])] for point in trajectory]
        bounds.extend(path)
        split_counts[split] += 1
        split_vins[split].add(vin)

        def optional_float(column: str) -> float | None:
            value = row.get(column)
            return None if value is None or pd.isna(value) else float(value)

        routes.append(
            {
                "trip_id": str(row["trip_id"]),
                "vehicle_id": pseudonymize_identifier(vin),
                "split": split,
                "split_label": _OVERVIEW_SPLITS[split]["label"],
                "color": _OVERVIEW_SPLITS[split]["color"],
                "start_time": str(row.get("StartTrip", "Unavailable")),
                "distance_km": optional_float("distance_km"),
                "duration_minutes": optional_float("duration_minutes"),
                "point_count": len(trajectory),
                "path": path,
            }
        )

    return {
        "routes": routes,
        "bounds": bounds,
        "trip_count": len(routes),
        "vehicle_count": int(trips["VIN"].astype(str).nunique()),
        "splits": {
            name: {
                **details,
                "trip_count": split_counts[name],
                "vehicle_count": len(split_vins[name]),
            }
            for name, details in _OVERVIEW_SPLITS.items()
        },
    }


def build_trip_overview_html(payload: Mapping[str, Any]) -> str:
    """Render all model-ready trips as a self-contained Leaflet map."""
    leaflet_css = (_ASSET_DIR / "leaflet.css").read_text(encoding="utf-8")
    leaflet_js = (_ASSET_DIR / "leaflet.js").read_text(encoding="utf-8")
    payload_json = json.dumps(payload, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>{leaflet_css}</style>
  <style>
    html, body, #vdp-trip-overview {{ width: 100%; height: 100%; margin: 0; background: #eef3f5; }}
    body {{ overflow: hidden; font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #vdp-trip-overview {{
      background-color: #edf2f4;
      background-image: linear-gradient(rgba(122,139,158,.08) 1px, transparent 1px), linear-gradient(90deg, rgba(122,139,158,.08) 1px, transparent 1px);
      background-size: 38px 38px;
    }}
    .leaflet-control-zoom a {{ color: #10233b; border-color: #d7dde5; }}
    .leaflet-control-layers {{ border: 1px solid #cfd7e0; border-radius: 6px; box-shadow: 0 2px 7px rgba(16,35,59,.12); }}
    .leaflet-control-layers-expanded {{ padding: 9px 11px; color: #27364a; font-size: 11px; line-height: 1.65; }}
    .leaflet-control-layers-selector {{ accent-color: #0968d8; margin-right: 5px; vertical-align: -1px; }}
    .leaflet-control-attribution {{ color: #68778a; font-size: 9px; }}
    .leaflet-tooltip {{ border: 1px solid #d7dde5; border-radius: 6px; box-shadow: 0 4px 12px rgba(16,35,59,.12); color: #27364a; font-size: 11px; line-height: 1.45; }}
    .vdp-coverage-summary {{ min-width: 164px; padding: 9px 11px; border: 1px solid #cfd7e0; border-radius: 6px; background: rgba(255,255,255,.95); box-shadow: 0 2px 7px rgba(16,35,59,.12); color: #27364a; font-size: 10px; line-height: 1.5; }}
    .vdp-coverage-summary strong {{ display: block; margin-bottom: 3px; color: #10233b; font-size: 11px; }}
    .vdp-swatch {{ display: inline-block; width: 8px; height: 3px; margin: 0 5px 2px 0; border-radius: 2px; }}
  </style>
</head>
<body>
  <div id="vdp-trip-overview" aria-label="All accepted model trips by dataset split"></div>
  <script>{leaflet_js}</script>
  <script>
    const data = {payload_json};
    const map = L.map('vdp-trip-overview', {{ zoomControl: true, preferCanvas: true, attributionControl: true }});
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      subdomains: 'abcd', maxZoom: 20,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);

    const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({{
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
    }}[character]));
    const groups = {{
      train: L.featureGroup(),
      validation: L.featureGroup(),
      test: L.featureGroup()
    }};
    const formatValue = (value, suffix) => value === null ? 'Unavailable' : `${{Number(value).toFixed(1)}}${{suffix}}`;

    data.routes.forEach((route) => {{
      const style = {{ color: route.color, weight: 3, opacity: .62, lineCap: 'round', lineJoin: 'round' }};
      const line = L.polyline(route.path.map((point) => [Number(point[1]), Number(point[0])]), style);
      const tooltip = `<strong>${{escapeHtml(route.vehicle_id)}}</strong><br>`
        + `${{escapeHtml(route.split_label)}} · ${{escapeHtml(route.start_time)}}<br>`
        + `${{formatValue(route.distance_km, ' km')}} · ${{formatValue(route.duration_minutes, ' min')}} · ${{route.point_count}} points`;
      line.bindTooltip(tooltip, {{ sticky: true }});
      line.on('mouseover', () => line.setStyle({{ weight: 5, opacity: 1 }}));
      line.on('mouseout', () => line.setStyle(style));
      line.addTo(groups[route.split]);
    }});

    const overlays = {{}};
    ['train', 'validation', 'test'].forEach((name) => {{
      const split = data.splits[name];
      const label = `${{split.label}} (${{split.trip_count}} trips · ${{split.vehicle_count}} vehicles)`;
      overlays[label] = groups[name];
      groups[name].addTo(map);
    }});
    L.control.layers(null, overlays, {{ collapsed: false, position: 'topright' }}).addTo(map);
    L.control.scale({{ imperial: false, position: 'bottomright' }}).addTo(map);

    const summary = L.control({{ position: 'bottomleft' }});
    summary.onAdd = () => {{
      const panel = L.DomUtil.create('div', 'vdp-coverage-summary');
      panel.innerHTML = `<strong>Model trip coverage</strong>${{data.trip_count}} accepted trips · ${{data.vehicle_count}} vehicles<br>`
        + ['train', 'validation', 'test'].map((name) => {{
          const split = data.splits[name];
          return `<span class="vdp-swatch" style="background:${{split.color}}"></span>${{split.label}}: ${{split.trip_count}}`;
        }}).join(' &nbsp; ');
      return panel;
    }};
    summary.addTo(map);

    const bounds = data.bounds.map((point) => [Number(point[1]), Number(point[0])]);
    if (bounds.length) map.fitBounds(bounds, {{ padding: [28, 28], maxZoom: 13 }});
    else map.setView([0, 0], 2);
    setTimeout(() => map.invalidateSize(), 0);
  </script>
</body>
</html>"""


def build_map_payload(
    trip: Mapping[str, Any],
    prediction: Mapping[str, Any],
    grid: GridEncoder,
    *,
    observed_prefix: list[list[float]] | None = None,
) -> dict[str, Any]:
    trajectory = parse_trajectory(trip["Trajectory"])
    prefix_fraction = float(prediction["prefix_fraction"])
    if observed_prefix is None:
        available = max(
            1,
            min(
                len(trajectory) - 1,
                int((len(trajectory) - 1) * prefix_fraction + 0.999),
            ),
        )
        prefix = trajectory[:available]
    else:
        prefix = [[float(point[0]), float(point[1])] for point in observed_prefix]
        if not prefix:
            raise ValueError("observed_prefix cannot be empty")
    candidates = json.loads(prediction["top_k_candidates"])
    for candidate in candidates:
        candidate["error_to_actual_km"] = haversine_km(
            candidate["latitude"],
            candidate["longitude"],
            prediction["actual_latitude"],
            prediction["actual_longitude"],
        )
        candidate["polygon"] = grid.polygon(candidate["cell"])
    points = [
        {
            "kind": "origin",
            "label": "Prediction origin",
            "latitude": float(prediction["origin_latitude"]),
            "longitude": float(prediction["origin_longitude"]),
            "color": [31, 41, 55, 230],
        },
        {
            "kind": "actual",
            "label": "Actual destination",
            "latitude": float(prediction["actual_latitude"]),
            "longitude": float(prediction["actual_longitude"]),
            "color": [16, 185, 129, 240],
        },
        {
            "kind": "refined",
            "label": f"Refined prediction ({float(prediction['error_km']):.2f} km error)",
            "latitude": float(prediction["predicted_latitude"]),
            "longitude": float(prediction["predicted_longitude"]),
            "color": [239, 68, 68, 240],
        },
    ]
    lines = [
        {
            "path": [
                [float(prediction["origin_longitude"]), float(prediction["origin_latitude"])],
                [float(prediction["actual_longitude"]), float(prediction["actual_latitude"])],
            ],
            "color": [16, 185, 129, 150],
        },
        {
            "path": [
                [float(prediction["origin_longitude"]), float(prediction["origin_latitude"])],
                [float(prediction["predicted_longitude"]), float(prediction["predicted_latitude"])],
            ],
            "color": [239, 68, 68, 150],
        },
    ]
    return {
        "trajectory": [[point[1], point[0]] for point in trajectory],
        "prefix": [[point[1], point[0]] for point in prefix],
        "points": points,
        "candidates": candidates,
        "lines": lines,
        "center": {
            "latitude": float(
                sum(point["latitude"] for point in points) / len(points)
            ),
            "longitude": float(
                sum(point["longitude"] for point in points) / len(points)
            ),
        },
    }


def build_leaflet_html(payload: Mapping[str, Any], visible_layers: set[str]) -> str:
    """Build a self-contained Leaflet canvas with only basemap tiles fetched remotely.

    Leaflet itself is bundled with the package, so the map controls and scientific
    overlays work even when a CDN is blocked. The optional light basemap needs
    ordinary internet access; without it, the overlays remain visible on a neutral
    geographic canvas.
    """

    leaflet_css = (_ASSET_DIR / "leaflet.css").read_text(encoding="utf-8")
    leaflet_js = (_ASSET_DIR / "leaflet.js").read_text(encoding="utf-8")
    payload_json = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    layers_json = json.dumps(sorted(visible_layers), separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>{leaflet_css}</style>
  <style>
    html, body, #vdp-map {{ width: 100%; height: 100%; margin: 0; background: #eef3f5; }}
    body {{ overflow: hidden; font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #vdp-map {{
      background-color: #edf2f4;
      background-image: linear-gradient(rgba(122,139,158,.08) 1px, transparent 1px), linear-gradient(90deg, rgba(122,139,158,.08) 1px, transparent 1px);
      background-size: 38px 38px;
    }}
    .leaflet-control-zoom a {{ color: #10233b; border-color: #d7dde5; }}
    .leaflet-control-layers {{ border: 1px solid #cfd7e0; border-radius: 6px; box-shadow: 0 2px 7px rgba(16,35,59,.12); }}
    .leaflet-control-layers-expanded {{ padding: 8px 10px; color: #27364a; font-size: 11px; line-height: 1.55; }}
    .leaflet-control-layers-selector {{ accent-color: #0968d8; margin-right: 5px; vertical-align: -1px; }}
    .leaflet-control-attribution {{ color: #68778a; font-size: 9px; }}
    .vdp-candidate {{
      width: 24px; height: 24px; display: flex; align-items: center; justify-content: center;
      border: 2px solid rgba(9,104,216,var(--opacity)); border-radius: 50%;
      background: rgba(255,255,255,.96); color: #075bbb; font-size: 11px; font-weight: 750;
      box-shadow: 0 1px 3px rgba(16,35,59,.18);
    }}
    .leaflet-tooltip {{ border: 1px solid #d7dde5; border-radius: 5px; box-shadow: 0 4px 12px rgba(16,35,59,.12); color: #27364a; font-size: 11px; }}
  </style>
</head>
<body>
  <div id="vdp-map" aria-label="Predicted and actual destination map"></div>
  <script>{leaflet_js}</script>
  <script>
    const data = {payload_json};
    const visible = new Set({layers_json});
    const map = L.map('vdp-map', {{ zoomControl: true, preferCanvas: true, attributionControl: true }});
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      subdomains: 'abcd', maxZoom: 20,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);
    const latLon = (path) => path.map((point) => [Number(point[1]), Number(point[0])]);

    const groups = {{
      'Trajectory': L.featureGroup(),
      'Observed prefix': L.featureGroup(),
      'Candidates (Top-K)': L.featureGroup(),
      'Actual & refined destinations': L.featureGroup(),
      'Candidate cells': L.featureGroup(),
      'Connection lines': L.featureGroup()
    }};
    L.polyline(latLon(data.trajectory), {{color:'#7b8da3', weight:3, opacity:.58}})
      .bindTooltip('Complete trip trajectory').addTo(groups['Trajectory']);
    L.polyline(latLon(data.prefix), {{color:'#0968d8', weight:5, opacity:.96}})
      .bindTooltip('Observed trajectory prefix').addTo(groups['Observed prefix']);
    data.lines.forEach((line) => {{
      const red = Number(line.color[0]) > Number(line.color[1]);
      L.polyline(latLon(line.path), {{color:red ? '#df3b3b' : '#15944a', weight:2, opacity:.48, dashArray:'6 7'}})
        .addTo(groups['Connection lines']);
    }});
    data.candidates.forEach((candidate) => {{
      L.polygon(latLon(candidate.polygon), {{color:'#0968d8', weight:1, opacity:.48, fillColor:'#0968d8', fillOpacity:.07}})
        .bindTooltip(`Cell ${{candidate.cell}}`).addTo(groups['Candidate cells']);
    }});
    const colors = {{origin:'#1f2937', actual:'#15944a', refined:'#df3b3b'}};
    data.points.forEach((point) => {{
      L.circleMarker([Number(point.latitude), Number(point.longitude)], {{
        radius: point.kind === 'origin' ? 8 : 9, color:'#ffffff', weight:3,
        fill:true, fillColor:colors[point.kind], fillOpacity:1
      }}).bindTooltip(point.label).addTo(groups['Actual & refined destinations']);
    }});
    data.candidates.forEach((candidate) => {{
      const rank = Number(candidate.rank);
      const opacity = Math.max(.48, 1 - rank * .055).toFixed(2);
      const icon = L.divIcon({{
        className:'', iconSize:[24,24], iconAnchor:[12,12],
        html:`<div class="vdp-candidate" style="--opacity:${{opacity}}">${{rank}}</div>`
      }});
      L.marker([Number(candidate.latitude), Number(candidate.longitude)], {{icon}})
        .bindTooltip(`Rank ${{rank}} · ${{(Number(candidate.probability)*100).toFixed(1)}}% · ${{Number(candidate.error_to_actual_km).toFixed(2)}} km · ${{candidate.cell}}`)
        .addTo(groups['Candidates (Top-K)']);
    }});

    const defaultKeys = {{
      'Trajectory':'Full trajectory', 'Observed prefix':'Observed prefix',
      'Candidates (Top-K)':'Candidates', 'Actual & refined destinations':'Destinations',
      'Candidate cells':'Candidate cells', 'Connection lines':'Connection lines'
    }};
    Object.entries(groups).forEach(([name, group]) => {{
      if (visible.has(defaultKeys[name])) group.addTo(map);
    }});
    L.control.layers(null, groups, {{collapsed:false, position:'topleft'}}).addTo(map);

    const bounds = latLon(data.trajectory);
    data.candidates.forEach((item) => bounds.push([Number(item.latitude), Number(item.longitude)]));
    if (bounds.length) map.fitBounds(bounds, {{padding:[28,28], maxZoom:14}});
    else map.setView([Number(data.center.latitude), Number(data.center.longitude)], 10);
    setTimeout(() => map.invalidateSize(), 0);
  </script>
</body>
</html>"""
