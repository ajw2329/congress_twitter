from sklearn.model_selection import StratifiedKFold
import pandas as pd


if __name__ == "__main__":

	all_tweets = pd.read_csv("../data/all_tweets_full.csv")
	wordvec_df = pd.read_csv("../data/all_tweets_wordvecs_twitter_model.csv")
	twitter_handles = pd.read_csv("../data/congress_twitter_handles_cleaned.csv")

	twitter_handles["kfold"] = -1 #instantiate kfold column

	twitter_handles = twitter_handles.sample(frac = 1).reset_index(drop = True)

	targets = twitter_handles.party.values == "R"

	kf = StratifiedKFold(n_splits = 6)

	for f, (t_, v_) in enumerate(kf.split(X = twitter_handles, y = targets)):

		twitter_handles.loc[v_, 'kfold'] = f


	## add kfold back to tweets dataframe

	name_kfold = twitter_handles[["name", "kfold"]]

	all_tweets = all_tweets.merge(name_kfold, on = "name", how = "left")

	all_tweets_train = all_tweets[~(all_tweets["kfold"] == 5)]

	all_tweets_test = all_tweets[all_tweets["kfold"] == 5]

	all_tweets_train.to_csv("../data/all_tweets_full_train.csv")
	all_tweets_test.to_csv("../data/all_tweets_full_test.csv")

	## add kfold back to wordvec dataframe

	wordvec_df = wordvec_df.merge(name_kfold, on = "name", how = "left")

	wordvec_df_train = wordvec_df[~(wordvec_df["kfold"] == 5)]

	wordvec_df_test = wordvec_df[wordvec_df["kfold"] == 5]

	wordvec_df_train.to_csv("../data/all_tweets_wordvecs_twitter_model_train.csv")
	wordvec_df_test.to_csv("../data/all_tweets_wordvecs_twitter_model_test.csv")