import pandas as pd
import json

def main():
    raw_data = pd.read_csv("data/raw.csv") # raw.csv 파일을 불러와 raw_data에 저장함
    minmax = raw_data.copy() 
    minmax_json = dict() #json 파일을 위해 빈 딕셔너리를 생성해줌

    for feature_name in raw_data.columns: # raw_data의 한 행씩 전처리를 진행함
        if feature_name == "Date": # Data column은 날짜에 관한 정보이기 때문에 전처리를 진행하지 않고 건너뜀
            continue
        else:
            max_value = float(raw_data[feature_name].max()) # 현재 column의 최대값과 최소값을 float형으로 구함
            min_value = float(raw_data[feature_name].min()) 
            # minmax 방식을 사용하여 데이터 전처리를 진행하고 혹시 모를 상황을 대비해 json 딕셔너리에 먼저 값을 저장해줌
            minmax[feature_name] = (raw_data[feature_name] - min_value) / (max_value - min_value)
            minmax_json[feature_name + "_max"] = max_value
            minmax_json[feature_name + "_min"] = min_value

    minmax.to_csv("data/minmax.csv", mode='w', index=False) 

    with open('data/minmax.json', 'w') as f:
        json.dump(minmax_json, f, indent="\t")
    f.close()
    
main()