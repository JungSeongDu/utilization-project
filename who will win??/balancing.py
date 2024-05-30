import pandas as pd
from sklearn.utils import resample

# 데이터 불러오기
df = pd.read_csv("study/data.csv")

# 각 클래스의 샘플 수 계산
red_count = df[df['winner'] == 'red'].shape[0]
blue_count = df[df['winner'] == 'blue'].shape[0]

print("Original red count:", red_count)
print("Original blue count:", blue_count)

# 클래스 간의 샘플 수 차이 계산
sample_diff = abs(red_count - blue_count)

# 클래스 간의 샘플 수가 다른 경우 샘플링하여 클래스 간의 샘플 수를 맞춤
if red_count > blue_count:
    df_red_downsampled = resample(df[df['winner'] == 'red'], replace=False, n_samples=blue_count, random_state=42)
    df_balanced = pd.concat([df_red_downsampled, df[df['winner'] == 'blue']])
elif blue_count > red_count:
    df_blue_downsampled = resample(df[df['winner'] == 'blue'], replace=False, n_samples=red_count, random_state=42)
    df_balanced = pd.concat([df[df['winner'] == 'red'], df_blue_downsampled])
else:
    df_balanced = df  # 클래스 간의 샘플 수가 같은 경우 그대로 사용

# 결과를 새로운 CSV 파일로 저장
print(df_balanced["winner"])


# 수정된 데이터프레임에서 클래스별로 샘플 수 출력
red_count_balanced = df_balanced[df_balanced['winner'] == 'red'].shape[0]
blue_count_balanced = df_balanced[df_balanced['winner'] == 'blue'].shape[0]

print("Balanced red count:", red_count_balanced)
print("Balanced blue count:", blue_count_balanced)