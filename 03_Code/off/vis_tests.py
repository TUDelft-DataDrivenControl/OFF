import matplotlib.pyplot as plt

# Define the x and y coordinates of the dots
x = [1, 2, 3, 4]
y = [5, 6, 7, 8]

# Define the x and y coordinates of the arrows
dx = [1, 2, -1, -2]
dy = [1, -1, 2, -2]

# Create a scatter plot of the dots
plt.scatter(x, y)

# Add arrows pointing from one dot to another
for i in range(len(x)):
    plt.arrow(x[i], y[i], dx[i], dy[i], length_includes_head=True, head_width=0.2)

# Show the plot
plt.show()