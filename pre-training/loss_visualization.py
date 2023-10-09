import matplotlib.pyplot as plt

# 8. Print loss graph
x = [i for i in range(0, len(loss_values_3))]
y_1 = loss_values_1
y_2 = loss_values_2
y_3 = loss_values_3
# Create a line plot for loss
plt.plot(x, y_1, marker='o', linestyle='-', color='blue', label='statistical-MLM')
plt.plot(x, y_2, marker='s', linestyle='--', color='green', label='dynamic-MLM')
plt.plot(x, y_3, marker='^', linestyle='-.', color='red', label='frequency-MLM')
# Adding labels and title
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Graph Over Epochs')
# Display the plot
plt.grid(True)
plt.show()
