[회고]

링크: https://colab.research.google.com/drive/1Gg6hozQqAMSsGKWdS4C5JNYeUT3pvCut?usp=sharing

기존의 성능 (Accuracy: 0.77)을 개선하기 위해서 Batch size와 Epochs를 각각 32와 15로 바꾸고,
좀 더 다양한 이미지 데이터를 사용하기 위해 이미지의 좌우를 랜덤하게 넣는 함수를 넣었다.
그리고 모델에 Drop-out과 Augmentation을 추가했지만 성능이 크게 늘지는 않았다 (Accuracy: 0.78).

기법을 많이 사용한다고 항상 좋은 결과가 나지 않는 것을 경험하면서
데이터의 종류, 모델의 특성을 고려하며 최고의 성능을 찾아가느 다양항 시도를 해보는 것이 중요하다는 것을 느낀다.
