import matplotlib.pyplot as plt
import numpy as np
import time

# Enable interactive mode
plt.ion()

# Initial setup: create a figure and axes
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')  # Initial line is empty
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Simulate a running algorithm that generates new data points
for i in range(10):
    x = np.linspace(0, i, 100)
    y = np.sin(x)
    
    # Update the data of the plot object
    line.set_data(x, y)
    
    # Adjust xlim if necessary
    ax.set_xlim(0, max(10, i + 1))
    
    # Redraw the plot
    plt.draw()
    plt.pause(0.1)  # Pause to update the plot and process GUI events

# Optional: turn off interactive mode and show the final plot
plt.ioff()
plt.show()
