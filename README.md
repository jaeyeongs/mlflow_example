# MLFlow 실습해보기

실무에서 데이터 분석, 머신러닝 & 딥러닝 모델 개발, 모델 배포까지 담당하고 있다. 그런데 모델 개발 시 모델의 성능 고도화를 위해 튜닝 작업을 하고 그에 따른 성능 비교를 실험적으로 반복하여 실시하는데 결과 지표를 일일이 기록해야 했다.

또한 모델 배포 시에는 직접 Django 프레임워크를 통해 API를 개발하여 POSTMAN으로 호출 결과를 확인한 후, Docker를 활용하여 모델 배포 패키징을 직접하였다. 배포 작업 또한 여러 작업이 필요한데, 이번에 새로운 프로젝트를 할 때 MLFlow를 도입하여 모델 개발 부터 배포까지 과정을 더 간결하면서 체계적으로 하고자 한다.

그러기 위해 미리 [MLFlow](https://github.com/jaeyeongs/research-develpoment/tree/main/ML/MLflow)가 무엇인지 사전 조사를 하고 [MLFlow를 활용한 MLOps](http://www.yes24.com/Product/Goods/106709982) 책으로 공부하면서 직접 실습을 해보았다. 

## 개발 환경

```
Numpy: 1.20.3
Pandas: 1.3.4
matplotlib: 3.4.3
seaborn: 0.11.2
scikit-learn: 0.24.2
MLFlow: 2.1.1
```

## MLFlow 실행

- 모델 학습 및 평가 코드를 작성한 후, MLFlow UI 실행은 Terminal에서 아래 명령어를 입력하면 된다.
- 기본 포트는 5000 이지만 원하는 포트로 설정 가능하다.

```python
mlflow ui -p 1234
```

### MLFlow 초기 화면

![mlflow_초기화면](https://user-images.githubusercontent.com/87981867/213847576-73198fe2-b97d-4c9c-9081-a1aaceb38ed8.png)

- 내가 설정한 'scikit_learn_experiment' 라는 실험에 들어가면 실험했던 모델의 결과가 간략히 나온다.

### 실험 결과

![mlflow_실험결과1](https://user-images.githubusercontent.com/87981867/213847702-cd9d12ea-b633-41c5-a0ea-2691330f98e5.png)

- 실험한 모델 하나를 선택해 들어가면 모델에 대한 파라미터, Metric 등을 확인할 수 있다.

### 실험 모델

![mlflow_실험모델](https://user-images.githubusercontent.com/87981867/213847758-24a08bcd-a672-41f2-a79d-893169d4f63b.png)

- 실험한 모델에 대한 모델 정보를 확인할 수 있다.
- requirements.txt, 모델 파일, 시각화 이미지 등 필요한 파일들이 생성 된다.

### 모델 비교

![mlflow_여러모델비교](https://user-images.githubusercontent.com/87981867/213847844-6dc45bad-3e1e-45ee-8c7f-611d46751b97.png)
![mlflow_여러모델비교2](https://user-images.githubusercontent.com/87981867/213847853-066be9b8-ddb6-4be5-a657-ddba9d77d6ed.png)

- 비교하고자 하는 모델을 선택하고 Compare 버튼을 누르면 선택한 모델을 table 형태로 비교해준다.
- 원하는 평가 지표를 통해 모델을 비교 할 수 있다.

## API Serving

- 배포하고자 하는 모델을 API Serving 하고 싶을 때는 아래 명령어를 입력하면 된다.
- 모델 ID와 지정한 모델 Name 두 가지 정보만 필요하다.

```python
mlflow models serve --model-uri runs:/bdfd5bcf28a64516bffd40a06bb5b2e8/log_reg_model -p 1235
```

## 실습 결론

모델 개발할 때 성능 비교를 하는 작업이 많이 필요한데 이런 경우에 굉장히 편리하다고 느꼈다. UI에서 편리하게 모델 관리가 가능한게 가장 큰 장점이라고 느꼈다. API 호출 또한 간단히 MLFlow 명령어로 가능해서 편했다.

MLOps를 위한 도구는 다양하지만, 머신러닝을 다루는 나에게는 MLFlow가 매우 유용한 도구로 쓰일 것 같다.

## 세미나 발표
- [MLOps를 위한 MLflow](https://github.com/jaeyeongs/mlflow_example/blob/main/seminar/MLOps%EB%A5%BC%20%EC%9C%84%ED%95%9C%20MLflow_%EC%8B%A0%EC%9E%AC%EC%98%81_230425.pdf)
