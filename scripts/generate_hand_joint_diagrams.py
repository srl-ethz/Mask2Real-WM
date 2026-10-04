"""Generate 23 static reference diagrams, one per action dimension, for the sine-actions
human rating study (scripts/gradio_sine_actions_rating.py).

Each diagram is a simple top-down schematic of the ORCA hand (not a URDF/sim render, which
avoids the Isaac Sim dependency and the mismatch between the real joint names and the URDF
joint names). The dimension's own joint/axis is
drawn in red and enlarged; every other joint is drawn small and gray. Dims 0-5 (wrist
pose x/y/z/roll/pitch/yaw) show an axis/arrow glyph at the wrist instead of a finger dot,
since they aren't a specific joint.

Usage:
    python scripts/generate_hand_joint_diagrams.py --output_dir assets/hand_joint_diagrams
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ACTION_LABELS = [
    "x", "y", "z", "roll", "pitch", "yaw",
    "wrist",
    "thumb_mcp", "thumb_abd", "thumb_pip", "thumb_dip",
    "index_abd", "index_mcp", "index_pip",
    "middle_abd", "middle_mcp", "middle_pip",
    "ring_abd", "ring_mcp", "ring_pip",
    "pinky_abd", "pinky_mcp", "pinky_pip",
]

DESCRIPTIONS = {
    "x": "Wrist position -- does the hand translate sideways (left/right)?",
    "y": "Wrist position -- does the hand translate forward/backward?",
    "z": "Wrist position -- does the hand translate up/down?",
    "roll": "Wrist orientation -- does the hand roll (rotate about its forward axis)?",
    "pitch": "Wrist orientation -- does the hand pitch (tilt up/down)?",
    "yaw": "Wrist orientation -- does the hand yaw (turn left/right)?",
    "wrist": "Hand's own wrist-flexion joint -- does the hand bend at the wrist (separate from the arm's wrist pose above)?",
    "thumb_mcp": "Thumb -- MCP joint, the knuckle where the thumb meets the palm.",
    "thumb_abd": "Thumb -- abduction, the thumb spreading sideways away from the palm.",
    "thumb_pip": "Thumb -- PIP joint, the middle knuckle of the thumb.",
    "thumb_dip": "Thumb -- DIP joint, the knuckle nearest the thumb tip.",
    "index_abd": "Index finger -- abduction, the finger spreading sideways at its base.",
    "index_mcp": "Index finger -- MCP joint, the main knuckle where the finger meets the palm.",
    "index_pip": "Index finger -- PIP joint, the middle knuckle.",
    "middle_abd": "Middle finger -- abduction, the finger spreading sideways at its base.",
    "middle_mcp": "Middle finger -- MCP joint, the main knuckle where the finger meets the palm.",
    "middle_pip": "Middle finger -- PIP joint, the middle knuckle.",
    "ring_abd": "Ring finger -- abduction, the finger spreading sideways at its base.",
    "ring_mcp": "Ring finger -- MCP joint, the main knuckle where the finger meets the palm.",
    "ring_pip": "Ring finger -- PIP joint, the middle knuckle.",
    "pinky_abd": "Pinky finger -- abduction, the finger spreading sideways at its base.",
    "pinky_mcp": "Pinky finger -- MCP joint, the main knuckle where the finger meets the palm.",
    "pinky_pip": "Pinky finger -- PIP joint, the middle knuckle.",
}

# (base_x, base_y, dir_x, dir_y, length, [joint_names_base_to_tip])
FINGERS = {
    "thumb": (-0.85, 0.05, -0.55, 0.55, 1.0, ["thumb_abd", "thumb_mcp", "thumb_pip", "thumb_dip"]),
    "index": (-0.45, 0.35, -0.12, 1.0, 1.05, ["index_abd", "index_mcp", "index_pip"]),
    "middle": (0.0, 0.4, 0.0, 1.0, 1.2, ["middle_abd", "middle_mcp", "middle_pip"]),
    "ring": (0.45, 0.35, 0.12, 1.0, 1.05, ["ring_abd", "ring_mcp", "ring_pip"]),
    "pinky": (0.8, 0.25, 0.28, 0.95, 0.85, ["pinky_abd", "pinky_mcp", "pinky_pip"]),
}
WRIST_XY = (0.0, -0.3)


def _joint_positions():
    """label -> (x, y) for every finger joint (wrist handled separately)."""
    positions = {}
    for base_x, base_y, dx, dy, length, joints in FINGERS.values():
        n = len(joints)
        for i, label in enumerate(joints):
            t = (i + 1) / n
            positions[label] = (base_x + dx * length * t, base_y + dy * length * t)
    return positions


def draw_hand(ax, highlight_label: str):
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.7, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")

    # Palm outline (rough trapezoid)
    palm = mpatches.Polygon(
        [(-0.9, -0.55), (0.9, -0.55), (1.0, 0.5), (-1.0, 0.5)],
        closed=True, facecolor="#f0e6d6", edgecolor="#999999", linewidth=1.5, zorder=1,
    )
    ax.add_patch(palm)

    # Fingers as line segments from base to tip
    for base_x, base_y, dx, dy, length, _ in FINGERS.values():
        tip_x, tip_y = base_x + dx * length, base_y + dy * length
        ax.plot([base_x, tip_x], [base_y, tip_y], color="#c7b299", linewidth=10,
                 solid_capstyle="round", zorder=2)

    joint_positions = _joint_positions()
    for label, (x, y) in joint_positions.items():
        is_target = label == highlight_label
        ax.plot(x, y, marker="o",
                 markersize=16 if is_target else 7,
                 markerfacecolor="#e0303a" if is_target else "#666666",
                 markeredgecolor="black" if is_target else "none",
                 markeredgewidth=1.5, zorder=3)

    # Wrist joint marker
    wx, wy = WRIST_XY
    is_wrist_target = highlight_label == "wrist"
    ax.plot(wx, wy, marker="s",
             markersize=16 if is_wrist_target else 9,
             markerfacecolor="#e0303a" if is_wrist_target else "#444444",
             markeredgecolor="black" if is_wrist_target else "none",
             markeredgewidth=1.5, zorder=3)

    # EE pose glyph (x/y/z/roll/pitch/yaw): arrow(s) below the wrist
    if highlight_label in ("x", "y", "z"):
        arrow_map = {"x": (1, 0), "y": (0, 1), "z": (0.4, 0.9)}
        adx, ady = arrow_map[highlight_label]
        ax.annotate("", xy=(wx + adx * 0.9, wy - 0.9 + ady * 0.9), xytext=(wx, wy - 0.9),
                    arrowprops=dict(arrowstyle="-|>", color="#e0303a", linewidth=4), zorder=4)
        ax.text(wx, wy - 1.05, f"translate ({highlight_label})", ha="center", fontsize=11, color="#e0303a")
    elif highlight_label in ("roll", "pitch", "yaw"):
        circle = mpatches.Arc((wx, wy - 0.75), 0.7, 0.7, angle=0, theta1=20, theta2=340,
                               color="#e0303a", linewidth=4, zorder=4)
        ax.add_patch(circle)
        ax.annotate("", xy=(wx + 0.32, wy - 1.05), xytext=(wx + 0.36, wy - 1.0),
                    arrowprops=dict(arrowstyle="-|>", color="#e0303a", linewidth=4), zorder=4)
        ax.text(wx, wy - 1.2, f"rotate ({highlight_label})", ha="center", fontsize=11, color="#e0303a")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="assets/hand_joint_diagrams")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for dim, label in enumerate(ACTION_LABELS):
        fig, ax = plt.subplots(figsize=(4.5, 5))
        draw_hand(ax, label)
        desc = DESCRIPTIONS[label]
        fig.suptitle(desc, fontsize=10, wrap=True, y=0.98)
        out_path = os.path.join(args.output_dir, f"dim_{dim:02d}_{label}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
