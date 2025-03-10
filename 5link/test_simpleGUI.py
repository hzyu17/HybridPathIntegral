import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from five_link_dynamics_new import *

class FiveLinkGUI:
    def __init__(self, master, x_trj):
        self.master = master
        master.title("5-Link Animation")

        # Create a matplotlib figure and axis
        self.fig, self.ax = plt.subplots(figsize=(20, 15))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a control frame for buttons
        control_frame = tk.Frame(master)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.animate_button = tk.Button(control_frame, text="Animate", command=self.start_animation)
        self.animate_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = tk.Button(control_frame, text="Stop", command=self.stop_animation)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.animating = False
        # Initialize configuration vector q (example with 7 elements)
        self.x_trj = x_trj
        self.t = 0
        self.nt = x_trj.shape[0]

    def start_animation(self):
        self.animating = True
        self.update_animation()

    def stop_animation(self):
        self.animating = False

    def update_animation(self):
        if not self.animating:
            return
        
        self.q = self.x_trj[self.t, 0:7]

        # Call draw_5link with the current configuration
        draw_5link(self.q, self.ax, legend=True)
        
        self.ax.figure.canvas.draw()

        self.t += 1
        
        if self.t == self.nt:
            self.t = 0
            
        # Schedule next update after 100 milliseconds
        self.master.after(10, self.update_animation)


if __name__ == "__main__":
    
    q_init = jnp.array([0, 0.658, 0, -0.6828+jnp.pi, 1.168, -0.6489+jnp.pi, 1.281])
    qdot_init = jnp.zeros(7)
    
    x_init = jnp.concatenate([q_init, qdot_init])
    
    fig, ax = plt.subplots()
    draw_5link(q_init, ax)
    plt.show()
    
    nt = 120
    dt_trj = np.ones(nt)*0.001
    nu = 4
    u_trj = np.zeros((nt, nu))
    u_trj[:30, 0] = 11 # u_1R
    u_trj[:30, 1] = -3 # u_2R
    u_trj[:30, 2] = -11 # u_1L
    u_trj[:30, 3] = -3.5 # u_2L

    u_trj[30:, 0] = -8 # u_1R
    u_trj[30:, 1] = 1 # u_2R
    u_trj[30:, 2] = 8 # u_1L
    u_trj[30:, 3] = 2.5 # u_2L
    
    
    x_trj = simulate_5link(x_init, u_trj, dt_trj)
    
    root = tk.Tk()
    app = FiveLinkGUI(root, x_trj)
    root.mainloop()
