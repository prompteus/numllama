import pandas as pd

def sanitize(pred):
    try:
        return int(pred)
    except:
        return -1

df = pd.read_csv('patching_results.csv', index_col=0).reset_index()
df['after_ok'] = df.after.apply(sanitize) == df.tgt
df_ = df[df.before.apply(sanitize) != df.tgt]
for grp, subdf in df_.groupby(['alpha', 'layer']):
    print(*grp, subdf.after_ok.mean())
# print(df.groupby('layer').after_ok.mean())

