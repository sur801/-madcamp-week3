# Mad Camp 3th week

**카이스트 몰입 캠프 3주차 프로젝트**  

**프로젝트 명** : 알파벳 수화 번역 프로그램 개발

**프로젝트 기간 :** 2020.07.23 ~ 2020.07.29

**개발 환경 :** Anaconda, Python, Tensorflow, OpenCV

</br>

## Sign Language Translator

- 프로젝트 내용
  - CNN (Convolution Neural NetWork)를 이용하여 알파벳 수화 이미지를 학습시킴.
  - Open Cv 의 web cam 을 통해 가져온 손 수화 이미지를 캡처 해와 알맞은 알파벳 수화로 classify 함. 
    그리고 classify된 알파벳을, web cam 화면 상의 black board에 띄움.
  - 여러 알파벳을 이용해 문장을 구성하고, TTS로 문장을 읽을 수 있게 함.
  - 학습 시킨 알파벳 수화 Image 는 11개로 A B C D E M O P I  / LOVE, MONEY(Custom Gesture) 가 있다.
    M과 A의 모양 자체가 비슷하여 인식을 정확히 하지 못하는 case가 더러 있음.



- 본인 역할
  - CNN 네트워크 학습
  - 학습시킬 손 수화 데이터 생성
  - 학습한 가중치를 이용해 알파벳 추론

</br>


## 각 파일들 상세 설명

<b>gesture.py</b> : 

- Open Cv Library의 web cam을 키고, 출력되는 카메라 화면에 수화를 표시할 박스(공간)을 만들어줌. <b>키보드로 p 버튼을 누르면</b>, 50프레임 동안 5프레임당 한번씩 박스에 해당하는 영역 만큼 10번 이미지를 가져와서 np array 형태로 저장함.
- numpy array로 만들어서 resizing 없이 grayscale로 전처리함.
  10개의 이미지를 np array형태로 바꾼 뒤, 미리 학습 시켜둔 CNN 모델 가중치를 불러와 알파벳 label을 predict하고, 10개의 이미지의 알파벳 label중 가장 빈도가 높은 label을 해당 gesture의 최종 label로 결정. 
  결정된 label에 해당하는 알파벳을 화면의 blackboard에 append해서 자막 처럼 출력함.
- 알파벳 수화로 문장을 만들기 위해 필요한 기능 : 
  - 키보드 s 입력 -> blackboard 에 출력되고 있는 문장 뒤에 공백(" ") 추가.
  - 키보드 d 입력 -> blackboard 에 출력되고 있는 문장 전체 삭제.
  - 키보드 b 입력 -> blackboard 에 출력되고 있는 문장 끝의 알파벳 삭제. backspace 기능.

- 문장 tts(text to speech) 를 위한 기능 :
  - 키보드 r 입력 -> blackboard에 표시된 문장을 tts 기능을 이용해 읽어줌.  

 

<b>make_gestures.py</b> :

 - web cam 상에 알파벳 수화를 표시할 박스(공간)을 그려줌. 키보드 p가 입력되면, 그 영역에서 2000 프레임 동안 1프레임씩 총 2000장의 이미지를 생성.
   이런식으로 한 알파벳 수화 당 총 8000 장의 학습 데이터를 생성함.  

 

<b>load_iamges.py</b> :

- 생성한 이미지들을 모두 불러와, image label을 추출하고 이미지들을 suffle해서 dump 파일로 저장함.  

 

<b>cnn_keras2.py</b> :

- CNN을 이용해 모델을 만들고 저장하는 파일.
- CNN은 4개의 layer로 이루어져 있는데, (2,2) pool_size와 3*3 convolution filter를 사용하였고, activation function은 relu와 softmax 를 
  사용하였다. 
- loss function은 categorical_crossentropy, optimizer는 adam을 
  사용했다.
- learning rate 와 epoch를 달리하면서 총 5개의 machine learning model를 만들어 보았고, 그중 epoch=10 learning rate=0.01, epoch=8 learning rate=0.005인 두가지 model이 test accuracy가 99% 이상으로 높게 나왔다. 그래서 이 두가지 모델을 사용하였다.  



</br>


## 실행 사진

 <img src="https://user-images.githubusercontent.com/5088280/100610507-34702900-3353-11eb-8b4d-7e2ba589dfab.png" width="400" height="270">  
