# coding:utf-8
import os, sys, time
import argparse
import requests
from bs4 import BeautifulSoup

def rem(str):
    """
    htmlのソースから画像リンクを抽出する
    """
    str0 = str.split('(')[1]
    return str0.split(';')[0]

def main():
    parser = argparse.ArgumentParser(description='Scrape line stamp')
    parser.add_argument('--base', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Base directory path to program files')
    parser.add_argument('--output_dir', type=str, default='../../data/raw',
                        help='Output directory')
    parser.add_argument('--label', type=int, default=10,
                        help='Label number (10: cat, 11: rabit, 12: dog, 13: bear, 14: bird, 19: panda, 20: seal)')
    args = parser.parse_args()

    # スタンプセットの番号
    setname = 0
    # 「カワイイ・キュート」ジャンルのスタンプを昇順に並べ、ページごとにデータを取得する
    for page in range(1, 11):
        ranking_url = 'https://store.line.me/stickershop/showcase/top_creators/ja?taste=1&character={}&page={}'.format(args.label, page)
        #requestsを使って、webから取得
        ran = requests.get(ranking_url)
        # 要素を抽出
        soup0 = BeautifulSoup(ran.text, 'lxml')
        # ソースの中でスタンプ一覧の箇所を探してリストに格納
        stamp_list = soup0.find_all(class_='mdCMN02Li')

        # スタンプセットループ
        for i in stamp_list:
            #スタンプセットに含まれる画像を表示させるページのリンク
            target_url = 'https://store.line.me{}'.format(i.a.get("href"))
            r = requests.get(target_url)
            output_dir = os.path.join(args.base, args.output_dir, '{0:04d}'.format(setname))
            os.makedirs(output_dir, exist_ok=True)
            setname += 1

            soup = BeautifulSoup(r.text, 'lxml') #要素を抽出
            span_list = soup.findAll("span",{"class":"mdCMN09Image"}) #スタンプセットに含まれる画像の情報をリストに格納

            # スタンプループ
            for stamp_number, i in enumerate(span_list):
                imgsrc = rem(i.get("style")) #画像データのURLを取得
                req = requests.get(imgsrc)

                if r.status_code == 200:
                    output_img = '{0:s}/{1:04d}.png'.format(output_dir, stamp_number)
                    with open(output_img, 'wb') as f:
                        f.write(req.content)

        print('finished downloading page: {} , set: ~{}'.format(page, setname))
        time.sleep(1)

if __name__ == '__main__':
    main()
