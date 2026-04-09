"""
src/generate_architecture_diagram.py
Update 11: Generate a high-resolution system architecture flowchart using matplotlib.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_diagram():
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Utility function to draw boxes
    def draw_box(x, y, w, h, text, color='#f8fafc', ec='#cbd5e1'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=3", fc=color, ec=ec, lw=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold', wrap=True)

    # 1. System Telemetry
    draw_box(40, 90, 20, 5, "System Telemetry\n(psutil, watchdog)")
    
    # Arrow to Feature Extractor
    ax.annotate("", xy=(50, 84), xytext=(50, 90), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    # 2. Feature Extractor
    draw_box(40, 78, 20, 6, "Feature Extractor\n(14 Metrics)")
    
    # Parallel lines to models
    ax.plot([50, 50], [78, 72], color='black', lw=1.5)
    ax.plot([10, 90], [72, 72], color='black', lw=1.5)
    
    # 3. Parallel Models
    model_xs = [5, 23, 41, 59, 77]
    models = ["Random\nForest", "XGBoost", "Calibrated\nSVM", "DNN\n(Keras)", "LSTM\n(Sequential)"]
    for x, name in zip(model_xs, models):
        ax.annotate("", xy=(x+9, 66), xytext=(x+9, 72), arrowprops=dict(arrowstyle="->", lw=1.5))
        draw_box(x, 60, 18, 6, name)
        
    # Lines from models to Meta-Ensemble
    for x in model_xs:
        ax.plot([x+9, x+9], [60, 55], color='black', lw=1.5)
    ax.plot([14, 86], [55, 55], color='black', lw=1.5)
    ax.annotate("", xy=(50, 48), xytext=(50, 55), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    # 4. Meta-Ensemble
    draw_box(35, 42, 30, 6, "Meta-Ensemble\n(Logistic Regression)")
    
    # Anti-Evasion Engine
    draw_box(5, 42, 20, 6, "Anti-Evasion\nEngine", color='#f0f9ff', ec='#2563EB')
    ax.annotate("Behavior Drift\nEntropy Analysis", xy=(35, 45), xytext=(25, 45), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 5. Threat Decision
    ax.annotate("", xy=(50, 32), xytext=(50, 42), arrowprops=dict(arrowstyle="->", lw=1.5))
    draw_box(35, 26, 30, 6, "Threat Decision\n(False Positive Reducer)", color='#fef2f2', ec='#DC2626')
    
    # 6. Response Engine
    ax.annotate("", xy=(50, 16), xytext=(50, 26), arrowprops=dict(arrowstyle="->", lw=1.5))
    draw_box(35, 10, 30, 6, "Response Engine\n(Mitigation Strategy)")
    
    # 7. Final Actions
    action_xs = [10, 40, 70]
    actions = ["Alert\n(Notification/Email)", "Quarantine\n(Process Suspend)", "Network Block\n(Firewall Rule)"]
    for x, act in zip(action_xs, actions):
        ax.annotate("", xy=(x+10, 4), xytext=(x+10, 10), arrowprops=dict(arrowstyle="->", lw=1.5))
        draw_box(x, -2, 20, 6, act, color='#ecfdf5', ec='#059669')

    plt.tight_layout()
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    save_path = os.path.join(reports_dir, "system_architecture.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Architecture diagram saved to {save_path}")

if __name__ == "__main__":
    generate_diagram()
