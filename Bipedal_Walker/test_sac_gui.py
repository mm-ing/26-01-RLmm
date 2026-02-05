"""
Quick GUI test for SAC to verify episodes are visible
"""

import tkinter as tk
from tkinter import ttk
from bipedal_walker_logic import BipedalWalkerEnvironment, SAC, Agent
import threading

class TestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAC Episode Test")
        self.root.geometry("600x400")
        
        # Create environment and policy
        self.env = BipedalWalkerEnvironment(hardcore=False)
        self.policy = SAC(24, 4, [64, 64], batch_size=64)
        self.agent = Agent(self.env, self.policy)
        
        # Status text
        self.status_text = tk.Text(root, height=20, width=70)
        self.status_text.pack(padx=10, pady=10)
        
        # Button
        tk.Button(root, text="Train 10 Episodes", command=self.train).pack(pady=5)
        
        self.training = False
        
    def log(self, msg):
        self.status_text.insert(tk.END, msg + "\n")
        self.status_text.see(tk.END)
        self.root.update()
        
    def train(self):
        if self.training:
            return
        
        self.training = True
        self.status_text.delete(1.0, tk.END)
        
        def train_thread():
            try:
                self.log("Training SAC for 10 episodes...")
                self.log("-" * 50)
                
                for ep in range(10):
                    reward = self.agent.run_episode(train=True)
                    memory = len(self.policy.memory)
                    
                    msg = f"Episode {ep+1}/10: Reward={reward:.2f}, Memory={memory}"
                    self.log(msg)
                    
                self.log("-" * 50)
                self.log("Training complete!")
                
            except Exception as e:
                self.log(f"Error: {str(e)}")
            finally:
                self.training = False
        
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = TestGUI(root)
    root.mainloop()
