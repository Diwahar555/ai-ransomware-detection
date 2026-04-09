"""
src/report_utils.py
Utilities for automated report content generation.
"""

def generate_limitations_section():
    """
    Update 12: Returns formatted text for the report limitations section.
    """
    limitations = [
        "1. Synthetic Dataset: The models were primarily trained on high-fidelity synthetic data. While statistically accurate, performance may vary on live, novel malware samples not present in the training distribution.",
        "2. Platform Dependency: The system is optimized for Windows environments. Behavioral telemetry (psutil/watchdog) behavior differs significantly on Linux or macOS, affecting detection consistency.",
        "3. Zero-Day Capability: While behavioral monitoring is robust, highly advanced zero-day threats using never-seen-before evasion techniques (e.g., kernel-level hooks) may bypass user-mode sensors.",
        "4. Sequential Stability (LSTM): The LSTM model may occasionally exhibit instability on certain hardware profiles, requiring an automated fallback to the DNN ensemble member.",
        "5. Network Isolation: Enabling automated network isolation may inadvertently affect legitimate background traffic or cloud synchronization services during a false-positive event.",
        "6. Whitelist Maintenance: As new professional software is installed, manual whitelist maintenance is required to ensure regular developer or system tools are not flagged as suspicious."
    ]
    
    formatted_text = "CHAPTER 8: SYSTEM LIMITATIONS\n" + ("=" * 30) + "\n\n"
    formatted_text += "The AI-Based Ransomware Pre-Attack Prediction System, while advanced, has the following technical and operational limitations:\n\n"
    for item in limitations:
        formatted_text += item + "\n\n"
        
    return formatted_text

if __name__ == "__main__":
    print(generate_limitations_section())
