import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/content/dog10_2_FNN_upd_mass_4_1709754743.8936841.csv')
# df.head()
plt.title('10 legged creature')
plt.plot(df['Step'], df['Value'], label='FNN')
leg = plt.legend(loc='upper left')