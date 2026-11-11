from .common import AccountData
from .cresci import load_cresci2017
from .cresci2015 import load_cresci2015
from .twibot import load_twibot20


def load(dataset: str, root: str, label_rule: str = "explicit", load_tweets: bool = False,
         max_tweets: int = 20) -> AccountData:
    if dataset == "cresci2017":
        return load_cresci2017(root, label_rule=label_rule, load_tweets=load_tweets,
                               max_tweets=max_tweets)
    if dataset == "cresci2015":
        return load_cresci2015(root, load_tweets=load_tweets, max_tweets=max_tweets)
    if dataset == "twibot20":
        return load_twibot20(root, load_tweets=load_tweets, max_tweets=max_tweets)
    raise ValueError(dataset)
