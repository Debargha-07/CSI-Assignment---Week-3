import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load
df = pd.read_csv("train.csv")

#Preview
print("First few rows of the dataset:")
print(df.head())

# Fill missing values 
df['Age'].fillna(df['Age'].median(), inplace=True)
df.dropna(subset=['Embarked'], inplace=True)

# Set style
sns.set(style="whitegrid")

#Bar Annotation
def annotate_bars(ax):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2, height + 2),
                    ha='center', fontsize=9,color='black')

# Create a figure 
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Titanic Dataset Visualization - Celebal Internship', fontsize=20, fontweight='bold')

# Plot 1: Survival Count
sns.countplot(data=df, x='Survived', ax=axs[0, 0], palette='Set2')
axs[0, 0].set_title('Overall Survival Count')
axs[0, 0].set_xticklabels(['Did Not Survive', 'Survived'])
axs[0, 0].set_xlabel('Survival Status')
axs[0, 0].set_ylabel('Passenger Count')

# Plot 2: Survival by Gender
sns.countplot(data=df, x='Sex', hue='Survived', ax=axs[0, 1], palette='pastel')
axs[0, 1].set_title('Survival by Gender')
axs[0, 1].legend(title='Survived', labels=['No', 'Yes'])
axs[0, 1].set_xlabel('Gender')
axs[0, 1].set_ylabel('Passenger Count')

# Plot 3: Age Distribution
sns.histplot(data=df, x='Age', bins=30, kde=True, ax=axs[1, 0], color='skyblue')
axs[1, 0].set_title('Age Distribution of Passengers')
axs[1, 0].set_xlabel('Age')
axs[1, 0].set_ylabel('Number of Passenger')

# Plot 4: Survival by Class
sns.countplot(data=df, x='Pclass', hue='Survived', ax=axs[1, 1], palette='muted')
axs[1, 1].set_title('Survival by Passenger Class')
axs[1, 1].set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
axs[1, 1].set_xlabel('Passenger Class')
axs[1, 1].set_ylabel('Passenger Count')

# Save figure 
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("titanic_visualization.png", dpi=300, bbox_inches='tight')  
plt.show()  

#Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()