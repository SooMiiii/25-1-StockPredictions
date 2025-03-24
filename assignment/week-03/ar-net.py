import torch
import torch.nn as nn
import torch.optim as optim

# ARNet 클래스: 단순 선형 회귀 기반의 자기회귀 모델
class ARNet(nn.Module):
    """
    ARNet (AutoRegressive Network)
    - 자기회귀 모델로, 과거 p개의 데이터를 입력으로 받아 한 개의 값을 예측하는 모델
    
    Attributes:
        linear (nn.Linear): p개의 입력 특징을 받아 1개의 출력을 생성하는 선형 계층
    """
    def __init__(self, p: int):
        """
        Args:
            p (int): 자기회귀 모델의 차수 (입력 데이터의 특징 개수)
        """
        super(ARNet, self).__init__()
        self.linear = nn.Linear(p, 1, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        모델의 순전파 연산 수행
        
        Args:
            x (torch.Tensor): (batch_size, p) 형태의 입력 데이터
        
        Returns:
            torch.Tensor: (batch_size, 1) 형태의 예측값
        """
        return self.linear(x)

    def sparse_total_prediction_error(self) -> torch.Tensor:
        """
        sTPE (Sparse Total Prediction Error) 계산
        
        Returns:
            torch.Tensor: sTPE 값 (0~100 사이의 값)
        """
        sTPE = 0
        for param in self.parameters():
            if len(param.shape) > 1:  # 가중치 파라미터에 대해서만 계산
                w_hat = param  # 모델 예측 가중치
                w = self.linear.weight  # 학습된 가중치
                numerator = torch.sum(torch.abs(w_hat - w))
                denominator = torch.sum(torch.abs(w_hat) + torch.abs(w))
                sTPE += 100 * numerator / denominator
        return sTPE

# SparseARNet 클래스: 가중치 희소성을 유도하는 정규화가 적용된 ARNet
class SparseARNet(ARNet):
    """
    SparseARNet
    - ARNet을 확장한 모델로, 정규화 항을 추가하여 가중치 희소성을 유도
    
    Attributes:
        c1 (float): 정규화 함수의 첫 번째 계수
        c2 (float): 정규화 함수의 두 번째 계수
        l1_lambda (float): L1 정규화 강도
    """
    def __init__(self, p: int, c1: float = 3.0, c2: float = 3.0, l1_lambda: float = 0.01):
        """
        Args:
            p (int): 자기회귀 모델의 차수
            c1 (float): 정규화 함수의 첫 번째 계수
            c2 (float): 정규화 함수의 두 번째 계수
            l1_lambda (float): L1 정규화 강도
        """
        super(SparseARNet, self).__init__(p)
        self.c1 = c1
        self.c2 = c2
        self.l1_lambda = l1_lambda
    
    def regularization(self) -> torch.Tensor:
        """
        R(θ): 정규화 항 계산
        
        Returns:
            torch.Tensor: 정규화 항 값
        """
        r_theta = 0
        for param in self.parameters():
            if len(param.shape) > 1:
                r_theta += torch.sum((2 / (1 + torch.exp(-self.c1 * torch.abs(param)**(1/self.c2)))) - 1)
        return r_theta

# 훈련 함수
def train(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 100, lr: float = 0.01, reg_lambda: float = 0.01):
    """
    모델을 주어진 데이터로 훈련시키는 함수
    
    Args:
        model (nn.Module): 훈련할 모델 (ARNet 또는 SparseARNet)
        X_train (torch.Tensor): (num_samples, p) 형태의 입력 데이터
        y_train (torch.Tensor): (num_samples, 1) 형태의 타겟 데이터
        epochs (int): 학습 반복 횟수
        lr (float): 학습률
        reg_lambda (float): SparseARNet의 정규화 강도
    """
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        if isinstance(model, SparseARNet):
            loss += reg_lambda * model.regularization()
        
        mse_loss = criterion(y_pred, y_train).item()
        sTPE_loss = model.sparse_total_prediction_error().item()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, MSE: {mse_loss:.4f}, sTPE: {sTPE_loss:.4f}')

# 메인 실행 함수
def main():
    """
    ARNet과 SparseARNet 모델을 랜덤 데이터로 학습시키는 메인 함수
    """
    p = 3  # 자기회귀 모델 차수
    X_train = torch.randn(1000, p)  # 입력 데이터 (랜덤)
    y_train = torch.randn(1000, 1)  # 타겟 값 (랜덤)
    
    print("훈련 시작: ARNet")
    ar_model = ARNet(p)
    train(ar_model, X_train, y_train, epochs=100, lr=0.01)
    print("ARNet 학습 완료\n")
    
    print("훈련 시작: SparseARNet")
    sparse_ar_model = SparseARNet(p)
    train(sparse_ar_model, X_train, y_train, epochs=100, lr=0.01)
    print("SparseARNet 학습 완료")

if __name__ == "__main__":
    main()