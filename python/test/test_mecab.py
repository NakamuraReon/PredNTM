import better_exceptions
import colored_traceback.always

import MeCab

text = "ティックトックでどんだけ食べてもゼロカロリーやってみた＼(^o^)／卍。"

pos_list = [10, 11, 31, 32, 34]
pos_list.extend(list(range(36,50)))
pos_list.extend([59, 60, 62, 67])
stop_words = ["する", "ない", "なる", "もう", "しよ", "でき", "なっ", "くっ", "やっ", "ある", "しれ", "思う", "今日", "それ", "これ", "あれ", "どれ", "どの", "NULL", "れる", "なり", "あっ"]

def create_mecab_list(text_list):
    mecab_list = []
    mecab = MeCab.Tagger("-d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd/dicrc")
    mecab.parse("")
    # encoding = text.encode('utf-8')
    for text in text_list:
        node = mecab.parseToNode(text)
        while node:
            # [品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用形,活用型,原形,読み,発音]
            # 忙しく  形容詞,自立,*,*,形容詞・イ段,連用テ接続,忙しい,イソガシク,イソガシク
            # morpheme = node.surface
            morpheme = " : ".join([node.surface, node.feature.split(",")[6], node.feature.split(",")[7]])
            if morpheme in stop_words:
                node = node.next
                continue
            if len(morpheme) > 0: # > 1:
                if node.posid in pos_list:
                    mecab_list.append(morpheme)
                    # print(morpheme, end=", ")
            node = node.next
    return mecab_list

mecab_list = create_mecab_list([text])

for w in mecab_list:
    print(w)