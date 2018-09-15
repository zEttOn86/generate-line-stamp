# coding:utf-8
import os, sys, time
import requests
from bs4 import BeautifulSoup

def rem(str):
    """
    htmlのソースから画像リンクを抽出する
    """
    str0 = str.split('(')[1]
    return str0.split(';')[0]

# スタンプセットのNo.(指定した番号から始まり、セットごとにフォルダを生成)
setname = 0
# 10: ネコ, 11:ウサギ, 12:イヌ, 13:クマ, 14:鳥, 19:パンダ, 20:アザラシ
chara = "&character=" + str(10)

# キャラクター指定せずにすべてを対象とする場合
#chara = ""

# 「カワイイ・キュート」ジャンルのスタンプを昇順に並べ、ページごとにデータを取得する
for page in range(1,11):
    ranking_url = 'https://store.line.me/stickershop/showcase/top_creators/ja?taste=1'+ str(chara) + '&page=' + str(page)
    #requestsを使って、webから取得
    ran = requests.get(ranking_url)
    # 要素を抽出
    soup0 = BeautifulSoup(ran.text, 'lxml')
    stamp_list = soup0.find_all(class_='mdCMN02Li') #ソースの中でスタンプ一覧の箇所を探してリストに格納

    for i in stamp_list:
        target_url = "https://store.line.me" + i.a.get("href") #スタンプセットに含まれる画像を表示させるページのリンク
        r = requests.get(target_url)         #requestsを使って、webから取得
        setname += 1
        #new_dir_path = str(setname) #スタンプセットのNo.に対応するフォルダを作成する
        #os.makedirs(new_dir_path, exist_ok=True) #フォルダが存在しない場合作成する

        soup = BeautifulSoup(r.text, 'lxml') #要素を抽出
        span_list = soup.findAll("span",{"class":"mdCMN09Image"}) #スタンプセットに含まれる画像の情報をリストに格納

        fname = 0 #ダウンロードする画像データの名称
        for i in span_list:
            fname += 1
            imgsrc = rem(i.get("style")) #画像データのURLを取得
            print(imgsrc)
            req = requests.get(imgsrc)

            #if r.status_code == 200:
            #    f = open( str(setname) + "/" + str(fname) + ".png", 'wb')
            #    f.write(req.content)
            #    f.close()

            # スクレイピングマナー
            time.sleep(1)

    print ("finished downloading page: " + str(page) + " , set: ~" + str(setname)  )
