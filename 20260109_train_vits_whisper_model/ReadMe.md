# Style-Bert-VITS2-改変

## 参考ソースコード
github：https://github.com/litagin02/Style-Bert-VITS2

## transcript修正

jvs_ver1/jvs070/nonpara30/transcripts_utf8.txt
BASIC5000_3078.wav ダム堤体中央部に、洪水吐を設置し、天端より、湖水の放流を行うものである。
                 ->ダム堤体中央部に、洪水ばきを設置し、天端より、湖水の放流を行うものである。

jvs_ver1/jvs070/nonpara30/transcripts_utf8.txt
BASIC5000_3026.wav 隔壁音波発生装置を、一定の周期で出しており、この隔壁音波を出している間は、鬼達の音撃を、無力化する。
                ->隔壁音波発生装置を、一定の周期で出しており、この隔壁音波を出している間は、鬼達のおんげきを、無力化する。
jvs_ver1/jvs100/nonpara30/transcripts_utf8.txt
BASIC5000_2036.wav 花乃、嘘泣きはたまーにやるから、効果あるんだぞ。
                 ->はなの、嘘泣きはたまーにやるから、効果あるんだぞ。

## TODO
lossの表記にdurationの平均二乗誤差とgeneratorの誤差を分けて描画するようにする

transcriptionの生成が煩雑
transcription.pyでjvsのtxt操作をしないといけない -> 自動化したい
text.py -> ファイル指定を毎回するのは面倒 config.yamlの操作かconfig.pyを変えて楽にしたい


## エラー
グォ が gw -> o にならないため gwでエラー
