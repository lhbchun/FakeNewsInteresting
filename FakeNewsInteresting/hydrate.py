def getUserLink(tweet):
    try:
        return tweet.user.link.url
    except AttributeError:
        return None

def tweetToDict(tweet):
    return {
            "id": tweet.id,
            "date": time.mktime(tweet.date.timetuple()), 
            "rawContent": tweet.rawContent,
            "replyCount": tweet.replyCount,
            "retweetCount": tweet.retweetCount,
            "likeCount": tweet.likeCount,
            "lang": tweet.lang,
            "source": tweet.source,
            "sourceLabel": tweet.sourceLabel,
            "viewCount": tweet.viewCount,
            "bookmarkCount": tweet.bookmarkCount,
            "pinned": tweet.pinned,

            "userId": tweet.user.id,
            "userDisplayName": tweet.user.displayname,
            "userRawDescription": tweet.user.rawDescription,
            "userVerified": tweet.user.verified,
            "userCreated": time.mktime(tweet.user.created.timetuple()),
            "userFollowersCount": tweet.user.followersCount,
            "userFriendsCount": tweet.user.friendsCount,
            "userStatusesCount": tweet.user.statusesCount,
            "userFavouritesCount": tweet.user.favouritesCount,
            "userListedCount": tweet.user.listedCount,
            "userMediaCount": tweet.user.mediaCount,
            "userLocation": tweet.user.location,
            "userProtected": tweet.user.protected,
            "userLink": getUserLink(tweet),
            "userBlue": tweet.user.blue,
            "userBlueType": tweet.user.blueType}


def saveTweet(directory, tweet):
    # with open("../../../Documents/PS928Local/testHydrated/1233160497756807172.json", "w") as outfile:
    with open( directory + str(tweet["id"]) + ".json", "w") as outfile:
        outfile.write(json.dumps(tweet))
        
def savePlaceholder(directory, fileName, text = "failed"):
    with open( directory + fileName, "w") as outfile:
        outfile.write(text)
        
def isTweetHydrated(directory, id):
    return os.path.exists(directory + str(id) + ".json")


def hydrateTweets(tweetList, directory, directoryIfFailed, retryFailed = False, batch = 1, pointer = 0):
    
    while pointer < len(tweetList):
        if isTweetHydrated(directoryIfFailed, tweetList[pointer]):
            print("Tweet hydration failed previously: " + str(tweetList[pointer]))
            if not retryFailed:
                pointer += batch
                continue

        if (not isTweetHydrated(directory, tweetList[pointer])):
            tweets = sntwitter.TwitterTweetScraper(tweetList[pointer])
            try:
                for i in tweets.get_items():
                    if type(i) != sntwitter.Tombstone:
                        thisTweet = tweetToDict(i)
                        saveTweet(directory, thisTweet)
                    else:
                        print("Note: Tweets Deleted for" + str(tweetList[pointer]))
                        savePlaceholder(directoryIfFailed, str(tweetList[pointer]) + ".json", "Deleted")
            except snscrape.base.ScraperException:
                print("Note: Tweets Result Error for" + str(tweetList[pointer]))
                savePlaceholder(directoryIfFailed, str(tweetList[pointer]) + ".json", "Not Scraped")
            except KeyError: # Some are deleted, rest are mostly politicians/news
                print("KeyError: " + str(tweetList[pointer]))
                savePlaceholder(directoryIfFailed, str(tweetList[pointer]) + ".json", "Unknown Key Error")
        else:
            print("Tweet already Hydrated: " + str(tweetList[pointer]))
        pointer += batch
        
        
def hydrateQuery(query, directory, directoryIfFailed = None, retryFailed = False):
    tweets = sntwitter.TwitterSearchScraper(query)
    for i in tweets.get_items():
        thisTweet = tweetToDict(i)
        saveTweet(directory, thisTweet)
        
        
def fetchOfficialCommunication(handle):
    month = 12
    year = 2019
    if not os.path.exists("../../../Documents/PS928Local/officalcomm/" + handle):
        os.makedirs("../../../Documents/PS928Local/officalcomm/" + handle)
        
    formatDate = datetime.date(year,month,1).strftime("%b%y")
    targetDir = "../../../Documents/PS928Local/officalcomm/" + handle + "/" + formatDate
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
        
    formateStart = str(year) + "-" + datetime.date(year,month,1).strftime("%m") + "-01"
    formateEnd = str(year) + "-" + datetime.date(year,month,1).strftime("%m") + "-" + str(calendar.monthrange(year, month)[1])
    query = "(from:" + handle + ") until:" + formateEnd +" since:" + formateStart +" -filter:replies"
    hydrateQuery(query, targetDir+ "/")
    
    for year in range(2020,2023):
        for month in range(1,13):
            formatDate = datetime.date(year,month,1).strftime("%b%y")
            targetDir = "../../../Documents/PS928Local/officalcomm/" + handle + "/" + formatDate
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            
            formateStart = str(year) + "-" + datetime.date(year,month,1).strftime("%m") + "-01"
            formateEnd = str(year) + "-" + datetime.date(year,month,1).strftime("%m") + "-" + str(calendar.monthrange(year, month)[1])
            query = "(from:" + handle + ") until:" + formateEnd +" since:" + formateStart +" -filter:replies"
            hydrateQuery(query, targetDir + "/")
    
        


# misc
# fetchOfficialCommunication("WHO") #Done
# fetchOfficialCommunication("NHSEngland") #Done
# fetchOfficialCommunication("NHSuk") #Done
# fetchOfficialCommunication("UNICEF") #Done
# fetchOfficialCommunication("UN") #Done
# fetchOfficialCommunication("ECDC_EU") #Done
# fetchOfficialCommunication("EU_Commission") #Done
# fetchOfficialCommunication("CDCgov") #Done
# fetchOfficialCommunication("CDCemergency") #Done
# fetchOfficialCommunication("CDCGlobal") #Done
# fetchOfficialCommunication("HHSGov") #Done
# fetchOfficialCommunication("DHSCgovuk") #Done
# fetchOfficialCommunication("PHE_uk") #Done 