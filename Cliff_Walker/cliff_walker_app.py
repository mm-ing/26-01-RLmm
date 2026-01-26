import tkinter as tk

from cliff_walker_gui import CliffWalkerGUI


def main():
    root = tk.Tk()
    app = CliffWalkerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
