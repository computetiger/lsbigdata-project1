import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import pandas as pd

# Dummy data for the pyramid plot
age_groups = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
male_population = [5, 8, 15, 20, 25, 22, 18, 10, 6, 3]
female_population = [6, 9, 14, 21, 24, 23, 17, 11, 7, 4]

# Pyramid plot using matplotlib
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(age_groups, male_population, color=plt.cm.Sunset(0.2), label='Male')
ax.barh(age_groups, [-x for x in female_population], color=plt.cm.Sunset(0.6), label='Female')
ax.set_xlabel('Population')
ax.set_ylabel('Age Groups')
ax.set_title('Population Pyramid')
ax.legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Dummy data for scatter plot
np.random.seed(42)
scatter_data = pd.DataFrame({
    'Year_Built': np.random.randint(1870, 2020, 50),
    'Year_Remod_Add': np.random.randint(1950, 2020, 50),
    'Branch_Number': np.random.randint(1, 100, 50)
})

# Scatter plot using plotly express
fig = px.scatter(scatter_data, x='Year_Built', y='Year_Remod_Add', size="Branch_Number", text="Branch_Number", 
                 color="Branch_Number", color_continuous_scale=px.colors.sequential.Sunset, size_max=60)

fig = fig.update_traces(textfont=dict(size=10), textposition='top center')

# Custom layout
fig = fig.update_layout(
    title='Year Built vs Remodel',
    xaxis_title='Year Built',
    yaxis_title='Year Remodel',
    xaxis=dict(range=[1870, 2020]),
    yaxis=dict(range=[1945, 2020])
)

fig.show()
