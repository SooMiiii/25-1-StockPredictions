# Week3

## 이번 주 과제
1. 데이터 전처리 • 시각화 (EDA)
2. 금융 데이터 처리 방법 조사 ( 평활화, 스케일링, 분해)
3. AR-Net 논문 읽고 구현 

## 폴더 구조
```
📂 week-03-ar-net/
├── 📄 README.md  (해당 주차 개요, AR-Net 구현 결과)
├── 📄 ar-net-review.md  (AR-Net 논문 리뷰)
└── 📄 ar-net.py  (AR-Net 구현)
```

## ARNet 및 SparseARNet 모델 설명
### ARNet 모델

`ARNet`은 선형 회귀 모델로, 주어진 입력 데이터 `X_train`에 대해 예측값 `y_pred`를 계산한다.

```python
class ARNet(nn.Module):
    def __init__(self, p):
        super(ARNet, self).__init__()
        self.linear = nn.Linear(p, 1, bias=True)

    def forward(self, x):
        return self.linear(x)
```
- `p`: 입력 특성의 수 (AR 차수)
- `self.linear`: 선형 변환 계층

### SparseARNet 모델
`SparseARNet`은 `ARNet`을 상속받아 가중치 정규화 항을 추가한 모델로, 희소성을 강화하는 정규화 함수 `R(θ)`를 사용한다.

```python
class SparseARNet(ARNet):
    def __init__(self, p, c1=3.0, c2=3.0, l1_lambda=0.01):
        super(SparseARNet, self).__init__(p)
        self.c1 = c1
        self.c2 = c2
        self.l1_lambda = l1_lambda

    def regularization(self):
        r_theta = 0
        for param in self.parameters():
            if len(param.shape) > 1:
                r_theta += torch.sum((2 / (1 + torch.exp(-self.c1 * torch.abs(param)**(1/self.c2)))) - 1)
        return r_theta
```

### sTPE 계산
`sTPE`는 예측 가중치와 실제 가중치 간 차이를 계산하여 모델 성능을 평가하는 지표

```python
def sparse_total_prediction_error(self):
    sTPE = 0
    for param in self.parameters():
        if len(param.shape) > 1:
            w_hat = param
            w = self.linear.weight
            numerator = torch.sum(torch.abs(w_hat - w))
            denominator = torch.sum(torch.abs(w_hat) + torch.abs(w))
            sTPE += 100 * numerator / denominator
    return sTPE
```

### train 함수
훈련 함수는 모델을 학습시키고, 각 에폭마다 MSE 및 sTPE 값을 출력
```python
def train(model, X_train, y_train, epochs=100, lr=0.01, reg_lambda=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        if isinstance(model, SparseARNet):
            loss += reg_lambda * model.regularization()

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
```

### main 함수
메인 함수에서 `ARNet`과 `SparseARNet` 모델을 훈련
```python
def main():
    p = 3
    X_train = torch.randn(1000, p)
    y_train = torch.randn(1000, 1)
    
    print("훈련 시작: ARNet")
    ar_model = ARNet(p)
    train(ar_model, X_train, y_train)
    
    print("훈련 시작: SparseARNet")
    sparse_ar_model = SparseARNet(p)
    train(sparse_ar_model, X_train, y_train)
```

### 결론
`python assignment/week-03/AR-Net.py` 의 결과
```
훈련 시작: ARNet
Epoch [0/100], Loss: 1.2164, MSE: 1.2164, sTPE: 0.0000
Epoch [10/100], Loss: 1.1484, MSE: 1.1484, sTPE: 0.0000
Epoch [20/100], Loss: 1.1021, MSE: 1.1021, sTPE: 0.0000
Epoch [30/100], Loss: 1.0706, MSE: 1.0706, sTPE: 0.0000
Epoch [40/100], Loss: 1.0491, MSE: 1.0491, sTPE: 0.0000
Epoch [50/100], Loss: 1.0345, MSE: 1.0345, sTPE: 0.0000
Epoch [60/100], Loss: 1.0245, MSE: 1.0245, sTPE: 0.0000
Epoch [70/100], Loss: 1.0177, MSE: 1.0177, sTPE: 0.0000
Epoch [80/100], Loss: 1.0130, MSE: 1.0130, sTPE: 0.0000
Epoch [90/100], Loss: 1.0099, MSE: 1.0099, sTPE: 0.0000
ARNet 학습 완료

훈련 시작: SparseARNet
Epoch [0/100], Loss: 1.4441, MSE: 1.4208, sTPE: 0.0000
Epoch [10/100], Loss: 1.3142, MSE: 1.2917, sTPE: 0.0000
Epoch [20/100], Loss: 1.2242, MSE: 1.2025, sTPE: 0.0000
Epoch [30/100], Loss: 1.1616, MSE: 1.1408, sTPE: 0.0000
Epoch [40/100], Loss: 1.1180, MSE: 1.0981, sTPE: 0.0000
Epoch [50/100], Loss: 1.0874, MSE: 1.0685, sTPE: 0.0000
Epoch [60/100], Loss: 1.0658, MSE: 1.0479, sTPE: 0.0000
Epoch [70/100], Loss: 1.0505, MSE: 1.0336, sTPE: 0.0000
Epoch [80/100], Loss: 1.0394, MSE: 1.0236, sTPE: 0.0000
Epoch [90/100], Loss: 1.0308, MSE: 1.0166, sTPE: 0.0000
SparseARNet 학습 완료
```

ARNet과 SparseARNet의 훈련 결과를 비교하면 다음과 같은 패턴이 나타난다.
- ARNet
    - MSE가 점진적으로 감소하여 1.0099에 수렴
    - sTPE 값은 모든 에폭에서 0.0000 유지
- SparseARNet
    - 초기 Loss는 ARNet보다 높지만 최종적으로 MSE 1.0166 도달
    - Loss와 MSE 차이는 정규화 항의 영향
    - sTPE 값이 0.0000으로 유지되어 희소성 유도가 충분히 작동하지 않았을 가능성 있음

랜덤 데이터를 사용했기 때문에 시계열 패턴이 없어 SparseARNet의 정규화 효과가 제대로 나타나지 않았다.

따라서 모델 성능을 정확히 비교하려면 실제 시계열 데이터를 사용하거나, 랜덤 데이터라도 자기상관 구조를 부여하는 것이 필요하다.

이번 실험은 AR-Net 및 SparseARNet의 기본적인 학습 과정을 확인하는 데 의미가 있으며, 정규화 효과를 제대로 분석하기 위해서는 데이터 구성의 중요성을 고려해야 한다.