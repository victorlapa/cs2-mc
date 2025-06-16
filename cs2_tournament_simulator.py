#!/usr/bin/env python3
"""
cs2_tournament_simulator_compare.py

Simula torneio de dupla eliminação (Major-style) comparando:
 1) Modelo NÃO-LINEAR: LightGBM
 2) Modelo LINEAR: LogisticRegression

Inclui otimizações para acelerar Monte-Carlo:
- Pré-cálculo de probabilidades de vitória por par de times
- Vetorização de séries com NumPy binomial
- Paralelização com joblib

Imprime resultados em Markdown sem depender de tabulate, e destaca líderes por métrica.

Uso:
    python cs2_tournament_simulator_compare.py \
      --size 8 --teams MIBR Vitality Astralis G2 Liquid Furia EG ENCE \
      --bo_wb 3 --bo_lb 2 --counts 10000 100000 1000000 --jobs 4
"""

import argparse  # parse comandos de terminal
import random    # seleção aleatória de times
import sys       # saída e erros
import warnings  # suprimir avisos
from itertools import combinations  # pares de times

import lightgbm as lgb           # modelo não-linear
import numpy as np               # operações numéricas
import pandas as pd              # manipulação de dados
from joblib import Parallel, delayed  # paralelização
from sklearn.compose import ColumnTransformer  # pré-processamento
from sklearn.impute import SimpleImputer       # imputação de valores
from sklearn.linear_model import LogisticRegression  # modelo linear
from sklearn.model_selection import train_test_split  # divisão treino/validação
from sklearn.pipeline import Pipeline             # pipeline de pré-processamento + modelo
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # codificação e padronização

# silencia avisos de usuário
warnings.filterwarnings("ignore", category=UserWarning)

# Função: converte DataFrame em tabela Markdown sem precisar de tabulate
# - Gera cabeçalho e linhas manualmente
# - Formata floats com 4 casas decimais

def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(['---'] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

# 1. Carrega dados históricos de partidas
# - Converte taxas de vitória em numérico
# - Extrai colunas de K/D, calcula médias e diferenciais
# - Retorna DataFrame filtrado e metadados de colunas KD

def load_data(path: str):
    df = pd.read_csv(path, sep=';')
    # winrates percentuais → floats
    df['T1_mapwinrate_num'] = df['T1_mapwinrate'].str.rstrip('%').astype(float)
    df['T2_mapwinrate_num'] = df['T2_mapwinrate'].str.rstrip('%').astype(float)
    # preenche modelos de K/D
    KD_TEMPLATE = "{}_player{}_K/D Ratio"
    KD_COLS_T1 = [KD_TEMPLATE.format("T1", i) for i in range(5)]
    KD_COLS_T2 = [KD_TEMPLATE.format("T2", i) for i in range(5)]
    KD_ALL = KD_COLS_T1 + KD_COLS_T2
    # coerção numérica e médias
    for c in KD_ALL:
        df[c] = pd.to_numeric(df.get(c, pd.Series()), errors='coerce')
    df['T1_kd_avg'] = df[KD_COLS_T1].mean(axis=1)
    df['T2_kd_avg'] = df[KD_COLS_T2].mean(axis=1)
    df['skill_diff'] = df['T1_mapwinrate_num'] - df['T2_mapwinrate_num']
    cols = ['team1','team2','map','team1_win',
            'T1_mapwinrate_num','T2_mapwinrate_num','skill_diff'] + KD_ALL + ['T1_kd_avg','T2_kd_avg']
    return df[cols].dropna(), KD_TEMPLATE, KD_COLS_T1, KD_COLS_T2, KD_ALL

# 2. Constrói pipelines de ML
# 2.1 NL: OneHot + StandardScaler + LightGBM
# 2.2 Lin: Impute numérico + LogisticRegression com features resumidos

def build_pipeline_nl(hist: pd.DataFrame) -> Pipeline:
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['team1','team2','map']),
        ('num', StandardScaler(), ['T1_mapwinrate_num','T2_mapwinrate_num','skill_diff','T1_kd_avg','T2_kd_avg'])
    ])
    return Pipeline([('pre', pre), ('clf', lgb.LGBMClassifier(objective='binary',n_estimators=700,random_state=42))])

def build_pipeline_lin(hist: pd.DataFrame):
    # média de map winrate por time (normalizada)
    m1 = hist[['team1','T1_mapwinrate_num']].rename(columns={'team1':'team','T1_mapwinrate_num':'mapwr'})
    m2 = hist[['team2','T2_mapwinrate_num']].rename(columns={'team2':'team','T2_mapwinrate_num':'mapwr'})
    map_means = pd.concat([m1,m2]).groupby('team')['mapwr'].mean()/100
    # média KD por time
    k1 = hist[['team1','T1_kd_avg']].rename(columns={'team1':'team','T1_kd_avg':'kd'})
    k2 = hist[['team2','T2_kd_avg']].rename(columns={'team2':'team','T2_kd_avg':'kd'})
    kd_means = pd.concat([k1,k2]).groupby('team')['kd'].mean()
    team_means = pd.DataFrame({'mapwinrate':map_means,'kd_avg':kd_means})
    # garante index para todos os times
    for t in pd.concat([hist['team1'],hist['team2']]).unique():
        team_means.loc[t] = team_means.get(t, {'mapwinrate':0.5,'kd_avg':1.0})
    # pipeline linear simples
    feat = ['T1_mapwr_norm','T2_mapwr_norm','skill_diff']
    df2 = hist.copy()
    df2['T1_mapwr_norm'] = df2['T1_mapwinrate_num']/100
    df2['T2_mapwr_norm'] = df2['T2_mapwinrate_num']/100
    df2['skill_diff']    = df2['T1_kd_avg'] - df2['T2_kd_avg']
    pre = ColumnTransformer([('num', SimpleImputer(strategy='median'), feat)])
    pipe = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=300,random_state=42))])
    pipe.fit(df2[feat], df2['team1_win'])
    return pipe, team_means

# 3. Pré-calcula probabilidades p(team1 vence) para cada par e mapa
#    usando os dois pipelines

def load_row_nl(t1,t2,map0,hist,KD_TEMPLATE,KD_COLS_T1,KD_COLS_T2):
    # extrai winrate histórico para o mapa ou geral
    def map_wr(tm):
        sel = pd.concat([
            hist[(hist['team1']==tm)&(hist['map']==map0)]['T1_mapwinrate_num'],
            hist[(hist['team2']==tm)&(hist['map']==map0)]['T2_mapwinrate_num']
        ])
        if sel.empty:
            sel = pd.concat([
                hist[hist['team1']==tm]['T1_mapwinrate_num'],
                hist[hist['team2']==tm]['T2_mapwinrate_num']
            ])
        return float(sel.mean())
    # média de KD
    def kd_avg(tm):
        return float(pd.concat([
            hist[hist['team1']==tm][KD_COLS_T1],
            hist[hist['team2']==tm][KD_COLS_T2]
        ]).stack().mean())
    base = {'team1':t1,'team2':t2,'map':map0}
    base.update({
        'T1_mapwinrate_num':map_wr(t1),'T2_mapwinrate_num':map_wr(t2),
        'skill_diff':map_wr(t1)-map_wr(t2),
        'T1_kd_avg':kd_avg(t1),'T2_kd_avg':kd_avg(t2)
    })
    # expande colunas K/D individuais
    for i in range(5):
        base[KD_TEMPLATE.format('T1',i)] = kd_avg(t1)
        base[KD_TEMPLATE.format('T2',i)] = kd_avg(t2)
    return pd.DataFrame([base])

def precompute_probs(teams,hist,pipe_nl,pipe_lin,map0,team_means,KD_TEMPLATE,KD_COLS_T1,KD_COLS_T2):
    probs_nl, probs_li = {},{}
    for t1,t2 in combinations(teams,2):
        df_nl = load_row_nl(t1,t2,map0,hist,KD_TEMPLATE,KD_COLS_T1,KD_COLS_T2)
        p_nl = pipe_nl.predict_proba(df_nl)[0,1]
        df_li = pd.DataFrame([{
            'T1_mapwr_norm': team_means.loc[t1,'mapwinrate'],
            'T2_mapwr_norm': team_means.loc[t2,'mapwinrate'],
            'skill_diff':    team_means.loc[t1,'kd_avg']-team_means.loc[t2,'kd_avg']
        }])
        p_li = pipe_lin.predict_proba(df_li)[0,1]
        probs_nl[(t1,t2)],probs_nl[(t2,t1)] = p_nl,1-p_nl
        probs_li[(t1,t2)],probs_li[(t2,t1)] = p_li,1-p_li
    return probs_nl,probs_li

# 4. Simula séries de mapas de forma vetorizada (Bernoulli)
#    retorna wins por série para t1 e t2

def simulate_series_vec(probs,pair,bo,size):
    p = probs[pair]
    if bo==1:
        w1 = np.random.binomial(1,p,size)
    else:
        w1 = np.random.binomial(bo,p,size)
        if bo==2:
            ties=(w1==1)
            w1[ties]=np.random.choice([0,2],ties.sum())
    return w1,bo-w1

# 5. Paraleliza Monte-Carlo de séries por par de times
#    contabiliza séries vencidas/derrotas, empates (BO2) e comebacks

def run_chunk(cnt,teams,probs,bo_wb,bo_lb):
    agg={t:{'win':0,'lose':0,'tie':0,'comeback':0} for t in teams}
    th=bo_wb//2+1  # gols necessários para vencer série
    for t1,t2 in combinations(teams,2):
        p=probs[(t1,t2)]
        first=np.random.binomial(1,p,cnt)
        rem=np.random.binomial(bo_wb-1,p,cnt) if bo_wb>1 else np.zeros(cnt,dtype=int)
        tot=first+rem
        win1=tot>=th
        win2=~win1
        if bo_wb==2:
            ties=(tot==1)
            agg[t1]['tie']+=ties.sum();agg[t2]['tie']+=ties.sum()
        agg[t1]['win']+=win1.sum();agg[t2]['lose']+=win1.sum()
        agg[t2]['win']+=win2.sum();agg[t1]['lose']+=win2.sum()
        agg[t1]['comeback']+=((first==0)&win1).sum()
        agg[t2]['comeback']+=((first==1)&win2).sum()
    return agg

# 6. Agrega resultados de todos os chunks

def combine(aggs,teams):
    total={t:{'win':0,'lose':0,'tie':0,'comeback':0} for t in teams}
    for a in aggs:
        for t in teams:
            for k in total[t]: total[t][k]+=a[t][k]
    return total

# 7. Fluxo principal: parse args, treina modelos, define times, executa MC

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--size',type=int,default=8)
    p.add_argument('--teams',nargs='*')
    p.add_argument('--random',action='store_true')
    p.add_argument('--bo_wb',type=int,default=3)
    p.add_argument('--bo_lb',type=int,default=2)
    p.add_argument('--counts',nargs='*',type=int,default=[10000,100000,1000000])
    p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--data',default='cs2_matches.csv')
    args=p.parse_args()

    # carrega e treina
    hist,KD_TEMPLATE,KD_COLS_T1,KD_COLS_T2,KD_ALL=load_data(args.data)
    pipe_nl=build_pipeline_nl(hist)
    X=hist.drop('team1_win',axis=1);y=hist['team1_win']
    Xtr,Xv,yt,yv=train_test_split(X,y,test_size=0.2,random_state=42)
    pipe_nl.fit(Xtr,yt)
    pipe_li,team_means=build_pipeline_lin(hist)
    map0=hist['map'].iloc[0]

    # seleciona times fixos ou aleatórios
    uniq=pd.concat([hist['team1'],hist['team2']]).unique()
    canon={t.lower():t for t in uniq}
    if args.teams:
        teams=[canon[t.lower()] for t in args.teams]
    elif args.random:
        teams=random.sample(list(uniq),args.size)
    else:
        sys.exit('Use --teams ou --random')

    # pré-computa probabilidades
    probs_nl,probs_li=precompute_probs(teams,hist,pipe_nl,pipe_li,map0,team_means,KD_TEMPLATE,KD_COLS_T1,KD_COLS_T2)

    # executa Monte-Carlo para cada contagem
    for n in args.counts:
        print(f"\n### Simulações: {n:,} torneios ###")
        chunks=[n//args.jobs]*args.jobs
        agg_nl=Parallel(n_jobs=args.jobs)(delayed(run_chunk)(c,teams,probs_nl,args.bo_wb,args.bo_lb) for c in chunks)
        final_nl=combine(agg_nl,teams)
        df_nl=pd.DataFrame(final_nl).T
        df_nl['win_rate']=df_nl['win']/(df_nl['win']+df_nl['lose']+df_nl['tie'])
        df_nl.sort_values('win',ascending=False,inplace=True)
        print("#### NÃO-LINEAR (LightGBM)")
        print(df_to_markdown(df_nl))
        # líderes com percentuais
        metrics=['win','lose','tie','comeback']
        totals={m:df_nl[m].sum() for m in metrics}
        print("🏆 Líderes NL:")
        for m in metrics:
            t=df_nl[m].idxmax();c=df_nl.loc[t,m];pct=c/totals[m]*100 if totals[m]>0 else 0
            print(f"  • Mais {m}: {t} ({int(c)}, {pct:.2f}%)")

        agg_li=Parallel(n_jobs=args.jobs)(delayed(run_chunk)(c,teams,probs_li,args.bo_wb,args.bo_lb) for c in chunks)
        final_li=combine(agg_li,teams)
        df_li=pd.DataFrame(final_li).T
        df_li['win_rate']=df_li['win']/(df_li['win']+df_li['lose']+df_li['tie'])
        df_li.sort_values('win',ascending=False,inplace=True)
        print("#### LINEAR (LogisticRegression)")
        print(df_to_markdown(df_li))
        print("🏆 Líderes LR:")
        for m in metrics:
            t=df_li[m].idxmax();c=df_li.loc[t,m];pct=c/df_li[m].sum()*100 if df_li[m].sum()>0 else 0
            print(f"  • Mais {m}: {t} ({int(c)}, {pct:.2f}%)")

if __name__=='__main__':
    main()
