

import snscrape.modules.twitter as sntwitter
#import csv
import pandas as pd
from random import uniform
from time import sleep
import pathlib


#maxTweets = 5000

#csvFile = open('place_result.csv', 'a', newline='', encoding='utf8')

#csvWriter = csv.writer(csvFile)
#csvWriter.writerow(['id','date','tweet',])
#
#for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:@SpeakerPelosi + since:2020-11-01 until:2020-11-05-filter:links -filter:replies').get_items()):
#	if i > maxTweets :
#		break
#csvWriter.writerow([tweet.id, tweet.date, tweet.content])
#csvFile.close()
#
#



def get_user_tweets(
    username, 
    start_date, 
    end_date, 
    max_tweets, 
    outdir, 
    sleep_sec, 
    **kwargs):

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
    outdir : str
        Path to output top directory.  Additional subdirectories will be created for
        the date ranges specified.
    **kwargs 
        Keyword arguments that specify additional column names and values to
        be added to the DataFrame


	Returns
	-------

	pandas.core.frame.DataFrame'
		Returns a pandas.core.frame.DataFrame containing data derived from
		snscrape.modules.twitter.Tweet objects
	'''

	username = username.replace('@', '')

    daterange_string = start_date + "_" + end_date
    filename = username + "_" + daterange_string + ".csv"
    filepath = pathlib.Path(outdir).joinpath(daterange_string, filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.is_file():

        print("Scraping tweets by user", username)

    	query = f'from:@{username} + since:{start_date} until:{end_date}-filter:links -filter:replies'

    	tweet_generator = sntwitter.TwitterSearchScraper(query).get_items()

        tweet_dict = {"username" : [], "date" : [], "tweet_content" : []}

        for tweet in tweet_generator:

            tweet_dict["username"].append(tweet.user.username)
            tweet_dict["date"].append(tweet.date)
            tweet_dict["tweet_content"].append(tweet.content)

        tweet_df = pd.DataFrame(tweet_dict)

        for col, val in kwargs.items():

            tweet_df[col] = val

        ## 

        tweet_df.to_csv(filepath)

        print("Scraped", tweet_df.shape[0], "tweets by user", username)

        sleep(uniform(sleep_sec, sleep_sec + 5))

    else:

        print("Tweets by user", username, "already scraped.  Reading file . . .")

        tweet_df = pd.read_csv(filepath)

        print("Read file containing", tweet_df.shape[0], "tweets by user", username)

    return tweet_df


def get_multi_user_tweets(
    usernames, start_date, end_date, max_tweets, outdir, sleep_sec, **kwargs):

    dfs = [None]*len(usernames)

    for i, user in enumerate(usernames):

        dfs[i] = get_user_tweets(
            user, start_date, end_date, max_tweets, outdir, sleep_sec, **kwargs)

    concat_df = pd.concat(dfs, axis = 0)

    return concat_df

def get_democrat_tweets(
    handles_df, start_date, end_date, max_tweets, outdir, sleep_sec):

    dem_handles = handles_df.Twitter[handles_df.Party.isin(["D", "I"])]

    dem_tweets = get_multi_user_tweets(
        dem_handles, start_date, end_date, max_tweets, outdir, sleep_sec, party = "D")

    return dem_tweets

def get_republican_tweets(
    handles_df, start_date, end_date, max_tweets, outdir, sleep_sec):

    rep_handles = handles_df.Twitter[handles_df.Party.eq("R")]

    rep_tweets = get_multi_user_tweets(
        rep_handles, start_date, end_date, max_tweets, outdir, sleep_sec, party = "R")

    return rep_tweets  


def get_all_tweets(handles_df, start_date, end_date, max_tweets, outdir, sleep_sec):

    dem_tweets = get_democrat_tweets(handles_df, start_date, end_date, max_tweets, outdir, sleep_sec)
    rep_tweets = get_republican_tweets(handles_df, start_date, end_date, max_tweets, outdir, sleep_sec)

    all_tweets = pd.concat([dem_tweets, rep_tweets], axis = 0)

    return all_tweets


handles_df = pd.read_csv("/home/aw/projects/congress_twitter/data/congress_twitter_handles.csv")


all_tweets = get_all_tweets(handles_df, "2020-01-01", "2020-12-31", max_tweets = 5000, outdir = "./output/", sleep_sec = 5)





#pelosi = get_user_tweets("SpeakerPelosi", "2020-01-01", "2020-12-31", 20000)



