import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_movies = 100
data = {
    'MovieTitle': [f'Movie {i}' for i in range(1, num_movies + 1)],
    'EarlyReviewSentiment': np.random.uniform(-1, 1, num_movies), # -1: very negative, 1: very positive
    'BoxOfficeRevenue': np.random.randint(1000000, 100000000, num_movies),
    'MarketingSpend': np.random.randint(100000, 10000000, num_movies),
    'Genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi'], num_movies)
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering ---
# No significant cleaning needed for synthetic data
# --- 3. Analysis ---
# Calculate correlation between sentiment and box office revenue
correlation = df['EarlyReviewSentiment'].corr(df['BoxOfficeRevenue'])
print(f"Correlation between Early Review Sentiment and Box Office Revenue: {correlation:.2f}")
# Simple linear regression (for demonstration)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = df[['EarlyReviewSentiment', 'MarketingSpend']]
y = df['BoxOfficeRevenue']
model.fit(X, y)
print(f"Linear Regression Coefficients: {model.coef_}")
print(f"Linear Regression Intercept: {model.intercept_}")
# --- 4. Visualization ---
# Scatter plot of sentiment vs. box office revenue
plt.figure(figsize=(8, 6))
sns.scatterplot(x='EarlyReviewSentiment', y='BoxOfficeRevenue', hue='Genre', data=df)
plt.title('Early Review Sentiment vs. Box Office Revenue')
plt.xlabel('Early Review Sentiment')
plt.ylabel('Box Office Revenue')
plt.savefig('sentiment_vs_revenue.png')
print("Plot saved to sentiment_vs_revenue.png")
# Box plot of box office revenue by genre
plt.figure(figsize=(10, 6))
sns.boxplot(x='Genre', y='BoxOfficeRevenue', data=df)
plt.title('Box Office Revenue by Genre')
plt.savefig('revenue_by_genre.png')
print("Plot saved to revenue_by_genre.png")
# Sentiment Analysis Example (using VADER)
analyzer = SentimentIntensityAnalyzer()
sample_review = "This movie was absolutely amazing! I loved every minute of it."
vs = analyzer.polarity_scores(sample_review)
print(f"\nSentiment Analysis of Sample Review: {vs}")