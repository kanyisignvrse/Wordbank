import pandas as pd

data = pd.read_csv('./NGSL_1.2_stats.csv')

articles = {"a", "an", "the"}

lemma_normalized = (
	data["Lemma"].astype(str).str.strip().str.lower()
)

cleaned = data[~lemma_normalized.isin(articles)]

cleaned.to_csv('./NGSL_1.2_stats_no_articles.csv', index=False)


