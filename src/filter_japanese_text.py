# Last Update: 2016-05-20
# @author: Satoshi Miyazawa
# koitaroh@gmail.com

# Applies simple text filter, then apply MeCab filter to break down text into words (separated by space)

from pyknp import Jumanpp
import MeCab
MECAB_MODE = 'mecabrc'
PARSE_TEXT_ENCODING = 'utf-8'

# Logging ver. 2016-07-12
from logging import handlers
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler('log.log', maxBytes=1000000, backupCount=3)  # file handler
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()  # console handler
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('Initializing %s', __name__)


def text_filter(text):
    # "RT @user:"を削除
    try:
        if "RT " in text and ":" in text:
            text = text.split(":", 1)[1]
            if text[0] == " ":
                text = text.lstrip(' ')
        # "@user"を削除
        #     if "@" in text and " " in text:
        if text[0] == "@":
            text = text.split(" ", text.count("@"))[-1]
            if text[0] == " ":
                text = text.lstrip(' ')
        # Delete tweets start with "I'm at ".
        if (text[0:7] == "I'm at ") or (text[0:7] == "I m at "):
            text = ""
            return text
        # "#tag"を削除
        if "#" in text:
            text = text.split("#", 1)[0]
        # "URL"を削除
        if "http" in text:
            text = text.split("http", 1)[0]
        # delete @ and following text (for Foursquare or Swarm)
        if "(@" in text:
            text = text.split("(@", 1)[0]
        # delete @ and following text (for other service)
        if "@" in text:
            text = text.split("@", 1)[0]
        text = text.replace('\n','') # Get rid of return
        text = text.replace('\r','') # Get rid of return
        text = text.replace("\"",' ') # Get rid of return
        text = text.replace("\(", ' ')  # Get rid of return
        text = text.replace("\)", ' ')  # Get rid of return
        text = text.replace("\'",' ') # Get rid of return
        text = text.replace("\\",' ') # Get rid of return
        text = text.rstrip()
    except IndexError:
        text = ""
    return text


def jumanpp_parse(text):
    jumanpp = Jumanpp()
    result = jumanpp.analysis(text)
    midashi_list = []
    if result:
        for mrph in result.mrph_list():
            midashi_list.append(mrph.midasi)
            # print(mrph.midasi)
    return midashi_list


def mecab_parse(text):
    tagger = MeCab.Tagger(MECAB_MODE)
    tagger.parse('')
    node = tagger.parseToNode(text)
    words = []
    nouns = []
    verbs = []
    adjs = []
    while node:
        pos = node.feature.split(",")[0]
        word = node.surface
        type(word)
        if pos == "名詞":
            nouns.append(word)
        elif pos == "動詞":
            lemma = node.feature.split(",")[6]
            # verbs.append(word)
            verbs.append(lemma)
        elif pos == "形容詞":
            lemma = node.feature.split(",")[6]
            # adjs.append(word)
            adjs.append(lemma)
        words.append(word)
        node = node.next
    parsed_words_dict = {
        "all": words[1:-1],
        "nouns": nouns,
        "verbs": verbs,
        "adjs": adjs
    }
    return parsed_words_dict


def dict_to_space_text_words(words_dict):
    words = " ".join(words_dict['all'])
    words = words.replace(',', ' ')
    return words


def list_to_space_text_words(words_list):
    words = " ".join(words_list)
    words = words.replace(',', ' ')
    return words


def filter_japanese_text_jumanpp(text):
    print(text)
    tweet_text = text_filter(text)
    print(tweet_text)
    words_list = jumanpp_parse(tweet_text)
    print(words_list)
    words = list_to_space_text_words(words_list)
    return words


def filter_japanese_text_mecab(text):
    print(text)
    tweet_text = text_filter(text)
    print(tweet_text)
    words_dict = mecab_parse(tweet_text)
    print(words_dict)
    words = dict_to_space_text_words(words_dict)
    return words

if __name__ == '__main__':
    # test_tweet = "@y1r16k微熱治ったけど喉が治らん(´･･｀)アンパンマンの顔作って～(笑)"
    test_tweet = "I m at 沼袋駅 (Numabukuro Sta.) (中野区, 東京都)"
    # test_tweet = "RT @test: I'm at アミューズメントパークエルロフト - @l_loft in 茨木市, 大阪府 https://t.co/Bgm813Qamu"
    # print(test_tweet)
    # tweet_text = text_filter(test_tweet)
    # print(tweet_text)
    # # words_dict = mecab_parse(tweet_text)
    # # words = dict_to_space_text_words(words_dict)
    # # logger.debug("Result run separately: %s", words)
    # # words2 = filter_japanese_text(test_tweet)
    # # logger.debug("Result run as one: %s", words2)
    #
    # words_list = jumanpp_parse(tweet_text)
    # print(words_list)
    # words = list_to_space_text_words(words_list)
    # print(words)

    # words2 = filter_japanese_text(test_tweet)
    # print(words2)

    words2 = filter_japanese_text_mecab(test_tweet)
    print(words2)