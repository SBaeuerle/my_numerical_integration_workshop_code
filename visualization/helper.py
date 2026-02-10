import os
import matplotlib.pyplot as plt

def smart_plot(fig, filename="plot_output.png"):
    """
    Handles static plot display.
    Saves to file if in Codespaces, otherwise shows a popup window.
    """
    if os.getenv('CODESPACES') == 'true':
        print(f"🌐 Cloud detected: Saving static plot to {filename}...")
        # dpi=300 ensures the text/lines look sharp when they open the file
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Success! Click '{filename}' in the sidebar to view.")
    else:
        print("💻 Local detected: Opening plot window...")
        plt.show()