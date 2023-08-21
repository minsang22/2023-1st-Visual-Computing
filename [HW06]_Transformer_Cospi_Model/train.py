import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime


def data_loader():
    csv_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/data/minmax.csv") # 전처리된 minmax.csv 데이터를 불러옴
    x = csv_data[["Close", "Open", "High", "Low", "Volume", "Change"]].to_numpy()
    y = csv_data[["Low"]].to_numpy() # y 값을 Low로 설정해서 코스피 지수의 최소값을 예측하도록 함

    dataX = []
    dataY = []

    testdataX = []
    testdataY = []

    for i in range(0, len(y) - 31):
        _x = x[i:i + 30] # 30일 동안의 데이터를 입력으로 사용하여 31일째 (다음날)의 값을 예측함
        _y = y[i + 31] # y 값에는 31일째의 데이터를 출력으로 사용함
        if i != len(y) - 32: 
            dataX.append(_x)
            dataY.append(_y)
        else: # 다음 날의 값을 예측하기 위해 따로 testdata에 저장해줌
            testdataX.append(_x)
            testdataY.append(_y)
    # numpy 배열을 Torch Tensor로 변환해줌
    dataX = torch.from_numpy(np.array(dataX))
    dataY = torch.from_numpy(np.array(dataY))
    testdataX = torch.from_numpy(np.array(testdataX))
    testdataY = torch.from_numpy(np.array(testdataY))

    return dataX, dataY, testdataX, testdataY


def train(model, dataX, dataY, epochs):
    model = model
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters())
    model.train()
    for epoch in range(epochs):
        pred = model(dataX.float().cuda()) # model에 입력 데이터를 전달하고 예측값을 얻음
        loss = criterion(pred, dataY.float().cuda()) # 예측값과 실제값 사이의 오차(loss)를 계산해줌
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("[ ", epoch + 1, " / ", epochs, "] loss : ", loss.item()) # 매 epoch마다 loss를 출력하여 확인해줌
    # torch.save(model.state_dict(), "./model/{}}.pt".format(datetime.today().date()))
    # 학습된 모델을 생성해주기 위한 코드이지만, 주석처리 해놓음

    
class Transformer(nn.Module): # Transformer 모델 정의
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                input_dim, 
                nhead=2,
                dim_feedforward=hidden_dim,
                dropout=0.2,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        out = self.fc(x[:, -1, :])
        return out


def main():
    dataX, dataY, testdataX, testdataY = data_loader()
    model = Transformer(input_dim=6, hidden_dim=6, output_dim=1, num_layers=2).cuda()
    train(model, dataX, dataY, 5000) # 실험 결과 epoch를 5000으로 설정하였을 때 오버피팅이 발생하지 않으며 가장 괜찮은 결과값이 나와 5000으로 설정하였음
    model.eval() # 최종 예측값을 확인하기 위한 eval() 함수 호출
    with torch.no_grad():
        prediction = model(testdataX.float().cuda())
    print("오늘의 최저값 : ", testdataY[0].float(), " / 내일의 예측 최저값 : ", prediction.float())
    # 코스피 지수 역변환 과정에서 계속 오류가 발생하여, 오늘의 최저값과 내일의 예측 최저값을 동시에 보여주며 둘을 비교할 수 있도록 하였음

    
main()