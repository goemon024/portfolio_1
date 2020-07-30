# 作成の背景と方向性（motivation.pdfを参照）。
### 情報収集のためにまとめサイトを使うことが多いが、時間効率の悪さを日頃から実感。
### そこで機械学習を利用して、まとめサイトの閲覧を効率化することを考えていく。
### アプリケーションの最終的なゴールとしては、「まとめサイトのカスタマイズや自動生成」のようなものを想定しており、開発を適宜進めていく。

### Portfolio_1について
### 　自分のインターネット閲覧履歴（まとめサイトクリック履歴）に基づいて、自分が選好するタイトルをＲＳＳからピックアップできるようにする。
　
### Portfolio_2について
### 　掲示板のタイトルだけでなく書込み内容も訓練データに含めて、閲覧価値のある掲示板をピックアップできるようにしたり、各種の指標（有用性、ニュースに対しての感情分析値、既存タイトルとの内容一致性など）を出力できるようにする。当面はＥＤＡを行いつつ教師なし学習の導入を検討する。
　
### ファイルの説明
###   motivation.pdf 上述
### 　sammary.pdf ポートフォリオ１の概要
### 　title_selection.py、vdata.py　実行ファイル
### 　title_selection.ipynb　ノートブックファイル
### 　title_scrayping.ipynb  データ収集用のスクレイピングファイル
### 
### 注１：
### 　モデルデータは容量の問題でGithubに上げていないので、そのままでは実行できません。
### 注２：
### 　トレーニングデータにおけるラベル０の収集は、上記のスクレイピングファイルを使用しましたが、ラベル１の収集はfirefoxの閲覧履歴を手動で取り込んで行いました。
