import pandas as pd
import matplotlib.pyplot as plt

# Load the results data from the CSV file
results_file_path = './result——300.csv'
results_data = pd.read_csv(results_file_path)

# Display the first few rows of the data to understand its structure
results_data.head()

# Create a plot of M vs C
plt.figure(figsize=(10, 6))
plt.plot(results_data['M'], results_data['C'], marker='o')
plt.title('Relationship between Memory Size M and Critical Threshold C')
plt.xlabel('Memory Size M')
plt.ylabel('Critical Threshold C')
plt.grid(True)
plt.show()