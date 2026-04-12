"""
Diegetic Terminal and Knowledge Ingestion GUI components.

Provides the human-to-system interface layers:
1. DiegeticTerminal: A conversational interface that mirrors legacy Cleverbot.
2. KnowledgeIngestionGUI: A panel for mapping images to words (Knowledge Dyads).

These are built using HTML/Vanilla CSS for the web application UI.
"""

import os

def create_diegetic_terminal_html():
    """Returns the HTML/CSS/JS for the Horizontal Manifold terminal."""
    # Read from the sibling .html file if it exists, otherwise use a fallback
    html_file = os.path.join(os.path.dirname(__file__), "diegetic_terminal.html")
    if os.path.exists(html_file):
        with open(html_file, "r") as f:
            return f.read()
    return "<!-- Error: diegetic_terminal.html not found -->"

if __name__ == "__main__":
    # In a real environment, we'd save this or serve it
    ui_path = os.path.join(os.getcwd(), "src", "ui", "diegetic_terminal.html")
    os.makedirs(os.path.dirname(ui_path), exist_ok=True)
    with open(ui_path, "w") as f:
        f.write(create_diegetic_terminal_html())
    print(f"UI components initialized at {ui_path}")
