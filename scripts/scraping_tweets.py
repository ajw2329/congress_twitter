

import snscrape.modules.twitter as sntwitter
import csv
import pandas as pd


maxTweets = 3000

csvFile = open('place_result.csv', 'a', newline='', encoding='utf8')

csvWriter = csv.writer(csvFile)
csvWriter.writerow(['id','date','tweet',])

for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:@SpeakerPelosi + since:2020-11-01 until:2020-11-05-filter:links -filter:replies').get_items()):
	if i > maxTweets :
		break
csvWriter.writerow([tweet.id, tweet.date, tweet.content])
csvFile.close()


handles_df = pd.read_csv("/home/aw/projects/congress_twitter/data/congress_twitter_handles.csv")



def get_user_tweets(username, start_date, end_date, max_tweets):

	'''
	Function that uses the snscrape module to scrape tweets
	given a username, date range, and maximum number of tweets to 
	consider.


	Parameters
	----------

	user : str
		Twitter username to query.  If '@' is prepended it will be removed by the function.
	start_date : str
		First date in the date range to search.  Expects YYYY-MM-DD format.
	end_date : str
		Last date in the date range to search.  Expects YYYY-MM-DD format.
	max_tweets : int
		Maximum number of tweets to return.

	Returns
	-------

	pandas.core.frame.DataFrame'
		Returns a pandas.core.frame.DataFrame containing data derived from
		snscrape.modules.twitter.Tweet objects
	'''

	username = username.replace('@', '')

	query = f'from:@{username} + since:{start_date} until:{end_date}-filter:links -filter:replies'

	tweet_generator = sntwitter.TwitterSearchScraper(query).get_items()

    tweet_dict = {"username" : [], "date" : [], "tweet_content" : []}

    for tweet in tweet_generator:

        tweet_dict["username"].append(tweet.user.username)
        tweet_dict["date"].append(tweet.date)
        tweet_dict["tweet_content"].append(tweet.content)

    return pd.DataFrame(tweet_dict)


get_user_tweets("SpeakerPelosi", "2020-11-01", "2020-11-10", 20)






