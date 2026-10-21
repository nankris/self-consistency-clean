"""Generate clean figures/diagram.excalidraw and render to PNG."""
import json
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

null = None

def create_element(id_name, el_type, x, y, w, h, **kwargs):
    el = {
        "id": id_name,
        "type": el_type,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": kwargs.get("strokeColor", "#1e1e1e"),
        "backgroundColor": kwargs.get("backgroundColor", "transparent"),
        "fillStyle": kwargs.get("fillStyle", "solid"),
        "strokeWidth": kwargs.get("strokeWidth", 2),
        "strokeStyle": kwargs.get("strokeStyle", "solid"),
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": null,
        "roundness": {"type": 3} if el_type == "rectangle" else None,
        "seed": abs(hash(id_name)) % 10000,
        "version": 1,
        "versionNonce": abs(hash(id_name)) % 10000 + 1,
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False
    }
    for k, v in kwargs.items():
        if k not in el:
            el[k] = v
    return el

def make_line_text(id_name, x, y, text, font_size=16, color="#1e1e1e"):
    w = max(len(text) * int(font_size * 0.65), 100)
    h = int(font_size * 1.3)
    return {
        "id": id_name,
        "type": "text",
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": null,
        "roundness": None,
        "seed": abs(hash(id_name)) % 10000,
        "version": 1,
        "versionNonce": abs(hash(id_name)) % 10000 + 1,
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
        "text": text,
        "fontSize": font_size,
        "fontFamily": 1,
        "textAlign": "left",
        "verticalAlign": "top",
        "containerId": null,
        "originalText": text,
        "autoResize": True,
        "lineHeight": 1.25,
        "baseline": font_size
    }

def make_arrow(id_name, start_x, start_y, end_x, end_y, stroke_color="#495057"):
    dx = end_x - start_x
    dy = end_y - start_y
    return {
        "id": id_name,
        "type": "arrow",
        "x": start_x,
        "y": start_y,
        "width": abs(dx),
        "height": abs(dy),
        "angle": 0,
        "strokeColor": stroke_color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": null,
        "roundness": {"type": 2},
        "seed": abs(hash(id_name)) % 10000,
        "version": 1,
        "versionNonce": abs(hash(id_name)) % 10000 + 1,
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
        "points": [[0, 0], [dx, dy]],
        "lastCommittedPoint": None,
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": "arrow"
    }

def main():
    elements = []
    
    # Header Titles
    elements.append(make_line_text("title", 40, 25, "Experiment 035b: Self-Consistency Done Right (Verification Pipeline)", 22, "#1864ab"))
    elements.append(make_line_text("subtitle", 40, 58, "Eliminating Prompt Formatting Artifacts and Validating Voting Gains on 500 Fresh GSM8K Items", 16, "#495057"))

    boxes = [
        {
            "id": "b1",
            "x": 40, "y": 105, "w": 290, "h": 240,
            "bg": "#ffe3e3", "border": "#c92a2a",
            "header": "1. THE PRIOR FLAW (035r5)",
            "header_color": "#c92a2a",
            "lines": [
                "• Prompt had 'Answer: NUMBER'",
                "• 1.7B copied literal 'NUMBER'",
                "• 55.75% parse failure rate",
                "• Primary +15.5 pp confounded",
                "  formatting fix with reasoning",
                "• Post-hoc true gain: +8.75 pp"
            ]
        },
        {
            "id": "b2",
            "x": 375, "y": 105, "w": 290, "h": 240,
            "bg": "#e7f5ff", "border": "#1971c2",
            "header": "2. CLEAN DESIGN & PILOT",
            "header_color": "#1864ab",
            "lines": [
                "• Prompt: 'The answer is [number]'",
                "• Robust multi-regex extractor",
                "• Normalizes decimals, commas",
                "• Pilot Gate (50 train items):",
                "  Require parse failure < 2.0%",
                "• 500 fresh unseen test items"
            ]
        },
        {
            "id": "b3",
            "x": 710, "y": 105, "w": 290, "h": 240,
            "bg": "#e6fcf5", "border": "#099268",
            "header": "3. CONFIRMATORY EVAL",
            "header_color": "#087f5b",
            "lines": [
                "• Model: Qwen3-1.7B (bfloat16)",
                "• Hardware: Colab L4 GPU",
                "• Greedy baseline: T=0, n=1",
                "• Stochastic: T=0.7, p=0.95, k=8",
                "• Evaluates k in {1, 3, 5, 8}",
                "• Plurality voting with tiebreak"
            ]
        },
        {
            "id": "b4",
            "x": 1045, "y": 105, "w": 290, "h": 240,
            "bg": "#fff9db", "border": "#f08c00",
            "header": "4. DECISION RULE",
            "header_color": "#d9480f",
            "lines": [
                "• 10,000 paired bootstrap CIs",
                "• Margin: +5.0 pp (8x compute)",
                "• PROVEN if:",
                "  Delta >= +5.0 pp AND",
                "  95% CI lower bound > +2.0 pp",
                "• DISPROVEN if 95% CI <= +5.0"
            ]
        }
    ]

    for b in boxes:
        # Rectangle
        elements.append(create_element(b["id"], "rectangle", b["x"], b["y"], b["w"], b["h"], backgroundColor=b["bg"], strokeColor=b["border"], strokeWidth=2))
        # Header
        elements.append(make_line_text(f"{b['id']}_h", b["x"] + 15, b["y"] + 15, b["header"], 16, b["header_color"]))
        # Content lines
        for i, line in enumerate(b["lines"]):
            elements.append(make_line_text(f"{b['id']}_l{i}", b["x"] + 15, b["y"] + 48 + i * 28, line, 14, "#212529"))

    # Arrows between boxes
    elements.append(make_arrow("arr1", 330, 225, 375, 225))
    elements.append(make_arrow("arr2", 665, 225, 710, 225))
    elements.append(make_arrow("arr3", 1000, 225, 1045, 225))

    diagram = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "viewBackgroundColor": "#ffffff",
            "gridSize": 20
        },
        "files": {}
    }

    excal_file = FIGURES_DIR / "diagram.excalidraw"
    excal_file.write_text(json.dumps(diagram, indent=2))
    print(f"Generated {excal_file}")

    png_file = FIGURES_DIR / "diagram.png"
    render_cmd = ["node", "/workspace/tools/excalidraw/render.cjs", str(excal_file), str(png_file), "2"]
    res = subprocess.run(render_cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Rendered {png_file} successfully!")
    else:
        print("Render failed:", res.stderr)

if __name__ == "__main__":
    main()
