import pandas as pd
import sys
from nltk.tokenize import TweetTokenizer
from collections import Counter

def lstcat(a, b):
    return a+b

def main():
    if len(sys.argv)==1:
        print("Please specify file")
        return
    df = pd.read_pickle(sys.argv[1])
    t = TweetTokenizer()
    df = pd.DataFrame({"month":df["created_at"].dt.month,"text":df["full_text"].str.lower(),"hashtags":df["hashtags"].apply(" ".join).str.lower()})
    df = pd.concat([df.groupby(["month"]).apply(lambda x: " ".join(x["hashtags"])).apply(t.tokenize),df.groupby(["month"]).apply(lambda x: " ".join(x["text"])).apply(t.tokenize)], axis=1)
    df.columns = ["hashtags","text"]
    df["hash_like_combination"] = (df.apply(lambda x: [i for i in x["text"] if i in x["hashtags"]], axis=1) + df["hashtags"]).apply(Counter).apply(lambda x: x.most_common(20))
    print(df)
    return


if __name__=="__main__":
    main()
