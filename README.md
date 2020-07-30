# まとめサイト閲覧の効率化について

### 【作成背景と方向性について（motivation.pdfを参照）】
  情報収集のために「まとめサイト」を使うことが多いが、時間効率が悪い　⇒　機械学習を利用してまとめサイト巡回の効率化を考える。  
  アプリケーションとしては、「まとめサイトのカスタマイズや自動生成」のようなものを最終的なゴールと考えて進めていく。  
### 【各ポートフォリオについて】
・Portfolio_1  
　自分のインターネット閲覧履歴（まとめサイトクリック履歴）に基づいて、自分が選好するタイトルをＲＳＳからピックアップできるようにする。  
・Portfolio_2  
　掲示板のタイトルだけでなく書込み内容も訓練データに含めるようにする。閲覧価値のある掲示板をピックアップできるようにしたり、各種の指標（有用性、ニュースに対しての感情分析値、既存タイトルとの内容一致性など）を出力できるようにする。当面はＥＤＡを行って教師なし学習の導入を検討する。
　
### 【Portfolio_1のファイルの説明】
motivation.pdf　・・・　上述  
sammary.pdf　・・・　ポートフォリオ１の概要  
title_selection.py、vdata.py　・・・　実行ファイル  
title_selection.ipynb　・・・　実行結果とコードが記載されたファイル    
title_scrayping.ipynb  ・・・　データ収集用のスクレイピングファイル  

###  【備考】
１：モデルデータは容量の問題でGithubに上げていないので、そのままでは実行できません。  
２：トレーニングデータにおけるラベル０の収集は上記のスクレイピングファイルを使用しましたが、ラベル１の収集はfirefoxの閲覧履歴を手動で取り込んで実行しています。
