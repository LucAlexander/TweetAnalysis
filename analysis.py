import pandas as pd
import sys
from nltk.tokenize import TweetTokenizer
from collections import Counter

if len(sys.argv)==1:
    print("Please specify file")
    exit()
df = pd.read_pickle(sys.argv[1])
t = TweetTokenizer()
df = pd.DataFrame({"month":df["created_at"].dt.month,"text":df["full_text"].str.lower(),"hashtags":df["hashtags"].apply(" ".join).str.lower()})
df = pd.concat([df.groupby(["month"]).apply(lambda x: " ".join(x["hashtags"])).apply(t.tokenize),df.groupby(["month"]).apply(lambda x: " ".join(x["text"])).apply(t.tokenize)], axis=1)
df.columns = ["hashtags","text"]
df["hlc"] = (df.apply(lambda x: [i for i in x["text"] if i in x["hashtags"]], axis=1) + df["hashtags"]).apply(Counter).apply(lambda x: x.most_common(20))
df["hlc"] = df["hlc"].apply(lambda x: [i[0] for i in x])
df = df.drop(columns=["text", "hashtags"])
df = pd.DataFrame({1:df.at[1, "hlc"],2:df.at[2, "hlc"],3:df.at[3, "hlc"],4:df.at[4, "hlc"],5:df.at[5, "hlc"],6:df.at[6, "hlc"],7:df.at[7, "hlc"]})
df.to_csv(sys.argv[1][:-1]+"csv", index=False)
