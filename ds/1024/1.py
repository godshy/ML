import pandas as pd

score = pd.read_csv('user_score.csv')
def above_mean(df, subject):

    user = len(score[score[subject] > score.describe()[subject]['mean']])

    # subjectは'kokugo','shakai','sugaku','rika'のいずれか

    return user
print(above_mean(score, 'kokugo'))

def score_sum(df):
    # 教科名は決め打ちでok
    df['sum'] = df.drop('user', axis=1).sum(axis=1)
    #df['sum'] = df[['kokugo', 'shakai', 'sugaku']].sum()
    return df

score = score_sum(score)
print(score.head())

def score_top3(df):
    score = df.sort_values(['sum'], ascending=False).head(3)
    score = list(score['user'])
    return score

print(score_top3(score))

