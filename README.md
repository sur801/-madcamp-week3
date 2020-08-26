# madcamp-week3
카이스트 몰입캠프 3주차. Machine learning. 알파벳 수화 번역 프로그램 개발

CNN (Convolution Neural NetWork)를 이용하여 이미지 학습시킴.
Open Cv Library를 web cam 기능을 이용해 화면 안에 수화를 인식 시킬 박스를 만들고, 그 박스 안의 이미지를 가져와서 처리함. Background의 image를 인식하고, 알파벳 수화를 표현하는 손의 threshold를 구분하여 0,255 scale의 이미지를 추출하는 방식과
100*100 frame안에서 찍은 사진을 그대로 grayscale로 바꾸어 학습시키는 방식으로 시도해봄.
결과적으로 threshold만 추출하는 방식은 손가락이 구분이 잘 되지 않아 이미지 인식률이 높지 않았음. 그래서 100*100 의 array를 numpy array로 만들어서 resizing 없이 grayscale로 전처리하는 방식을 선택함.




CNN은 4개의 layer로 이루어져 있는데, image size에 맞는 pool_size와 convol size을 사용하였고, activation func과 같은 구체적인 값들은 
cnn_keras*.py file에서 찾아볼 수 있다. 

learning rate 와 epoch를 달리하면서 총 5개의 machine learning model를 만들어 보았고,
그중 epoch=10 learning rate=0.01, epoch=8 learning rate=0.005인 두가지 model이 test accuracy가 99% 이상으로 높게 나왔다. 그래서 이 두가지 모델을 사용하였다.

학습 시킨 알파벳 수화 image 는 11개로 A B C D E M O P I  LOVE MONEY(custom gesture) 가 있다.
M과 A의 모양 자체가 비슷하여 인식을 정확히 하지 못하는 case가 더러 있다. 
