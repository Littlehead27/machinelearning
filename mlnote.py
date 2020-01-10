#神奇的axis=0/1 : 
#合并的时候，axis=0代表rbind，axis=1代表cbind；
#单个dataframe时候，axis=0代表列，axis=1代表行

#-------------------------------EDA-------------------------------------
df.info()
df.dtypes
#空值判斷
df.isnull().sum(axis=0)
#查看統計指標
df.describe(include='all').T
# 设置显示最大字段数后，可显示全部字段：
pd.set_option('display.max_columns', 60)

features = [f for f in train_data.columns if f not in ['用户编码','信用分']]
for f in features:
    print(f + "的特征分布如下：")
    print(train_data[f].describe())
    if train_data[f].nunique()<20:
        print(train_data[f].value_counts())
    plt.hist(train_data[f], bins=70)
    plt.show()

# 缺失值分析
def missing_values(df):
    alldata_na = pd.DataFrame(df.isnull().sum(), columns={'missingNum'})
    alldata_na['existNum'] = len(df) - alldata_na['missingNum']
    alldata_na['sum'] = len(df)
    alldata_na['missingRatio'] = alldata_na['missingNum']/len(df)*100
    alldata_na['dtype'] = df.dtypes
    #ascending：默认True升序排列；False降序排列
    alldata_na = alldata_na[alldata_na['missingNum']>0].reset_index().sort_values(by=['missingNum','index'],ascending=[False,True])
    alldata_na.set_index('index',inplace=True)
    return alldata_na
missing_values(data_train)

# 特征nunique分布
# communityName值太多，暂且不看图表
features = [f for f in categorical_feas if f not in ['communityName']]
for feature in features:
    print(feature + "的特征分布如下：")
    print(data_train[feature].value_counts())
    plt.hist(data_all[feature], bins=3)
    plt.show()

# 统计特征值出现频次大于100的特征
for feature in categorical_feas:
    df_value_counts = pd.DataFrame(data_train[feature].value_counts())
    df_value_counts = df_value_counts.reset_index()
    df_value_counts.columns = [feature, 'counts'] # change column names
    print(df_value_counts[df_value_counts['counts'] >= 100])

# 数值型特征看下相关性，比赛时一般相关性超过95或98%才会删掉
from scipy.stats import pearsonr
for i in range(0, len(numerical_feas)-1):
    for j in range(i+1, len(numerical_feas)):
        f1 = numerical_feas[i]
        f2 = numerical_feas[j]
        if abs(data_train[[f1,f2]].corr().values[0][1]) > 0.90:
            print(f1, f2, data_train[[f1,f2]].corr().values[0][1])

#-------------------------------异常值处理-------------------------------------
data.loc[data['local_dur'] > 200000, 'local_dur'] = 200000
# 处理离群点
q1 = submission['SalePrice'].quantile(0.0045) #上分位数
q2 = submission['SalePrice'].quantile(0.99)   #下分位数
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

# 孤立森林删除异常值
# 实验表明，当设定为 100 棵树，抽样样本数为 256 条时候，IF 在大多数情况下就已经可以取得不错的效果
from sklearn.ensemble import IsolationForest
def IF_drop(train):
    rng = np.random.RandomState(42)
    IForest = IsolationForest(max_samples=256,random_state=rng)
    IForest.fit(train["tradeMoney"].values.reshape(-1,1))
    y_pred = IForest.predict(train["tradeMoney"].values.reshape(-1,1))
    drop_index = train.loc[y_pred==-1].index
    print(drop_index)
    train.drop(drop_index,inplace=True)
    return train
data_train = IF_drop(data_train)

#-------------------------------缺失值处理-------------------------------------
# 缺失值过滤 删除控制比例过大的列
a = df.isnull().sum()/len(df)*100
variables = df.columns #将列名保存在变量中
variable = []
for i in range(0,len(df.columns)):
    if a[i]<=10: #将阈值设置为10%
        variable.append(variables[i])
df=df[variable]
#  按阈值来删除
# df=df.dropna(thresh=2000,axis=1) # thresh 非空个数 

#填补众位数
from scipy.stats import mode
data['Self_Employed'].fillna(mode(data['Self_Employed']).mode[0], inplace=True)
#sklearn
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
#先分组再填充
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
#特定字符填充
dataset['Cabin'] = dataset['Cabin'].fillna('U')
#中位数填充
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
#平均值填充
for x in data.columns:
    data[x].fillna(data[x].mean(), inplace=True)
#如果出现正最大及负最大值，类似处理
data = data.replace(-np.inf, np.nan)
data = data.replace(np.inf, np.nan)
data.fillna(0, inplace = True)

#-------------------------------dataframe各种操作-------------------------------
#1 – 布尔索引
data.loc[(data["Gender"]=="Female") & (data["Education"]=="Not Graduate") & (data["Loan_Status"]=="Y"),
    ["Gender","Education","Loan_Status"]]
#apply函数
data.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column
data.apply(num_missing, axis=1) #axis=1 defines that function is to be applied on each row
data['disk_read_kb'] = data['disk_read_kb'].apply(lambda x: 400 if x>=400 else x)
#排序
data.sort_values(['ApplicantIncome','CoapplicantIncome'], ascending=False)
#删除列
data.drop(['complete_code_id'], axis=1, inplace=True)
#更改列名
data.rename(columns={'accept_id_x':'accept_id', 'complain_nbr_x':'complain_nbr'}, inplace = True)
#查询每个类别的记录数
train['accept_org_id'].value_counts().plot(kind='bar')
#获取时间
data['accept_hour'] = data['accept_date'].apply(lambda x: datetime.strptime(x, '%H%M%S').hour)
#从unix时间戳转换时间
data['startTime'] = data['startTime'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000000).strftime("%Y-%m-%d %H:%M"))
#转换时间
data['update_time'] = data['update_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M'))
#获取离该记录最近的那条记录时间
alarm['update_time2'] = alarm.apply(lambda row: data.loc[(data.host == row['host']) 
                                                         & (data.update_time <= row['update_time']), 'update_time'].max(), axis=1)
#根据时间序列来关联
data['timedet_1'] = data['update_time'].shift(1) #平移一行，要先排好序号
data = pd.merge(data, data, how='left', left_on=['host', 'timedet_1'], right_on=['host', 'update_time'], suffixes=('', '_1'))
data['timedet_1'] = (data['update_time'] - data['timedet_1']).apply(lambda x: x.total_seconds()) # 两个时间的相差秒数
#跟前5分钟的数据关联
data['endTime_5'] = data['endTime'] - datetime.timedelta(minutes=5)
data = pd.merge(data, data, how='left', left_on=['host', 'endTime'],right_on=['host', 'endTime_5'], suffixes=('', '_5'))
#归并文字内容进行归并
alarm['alarm'] = alarm['alarm'].apply(lambda x: '文件系统使用率告警' if x.startswith('文件系统使用率告警') else x)
alarm['alarm'] = alarm['alarm'].apply(lambda x: '采集网元数据失败' if x.find('采集网元数据失败')>=0 else x)
#重置索引
train = train.reset_index(drop=True)
#去重
data = data.drop_duplicates(['host', 'update_time'], keep='last')
#特征编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_to_encode = ['accept_org_id','serv_item']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])
#关联两张表，suffixes指定关联键的后缀名
data = pd.merge(data, errDict, how='left', left_on='complete_code', right_on='complete_code_id', suffixes=('', '_y'))
data = pd.merge(data, df, left_on=['area_code','exch_id','sub_serv_type','accept_hour'],
                right_index=True)  # 如果一个是用列关联，一个是用索引关联，要指定 right_index 或者 left_index
#group by 行数和唯一数
df = data.groupby(['area_code','exch_id','sub_serv_type','accept_hour']).agg({'accept_id':'count', 'complain_nbr':'nunique'})
#group by 后排序
df = train.groupby(['accept_org_id'])[['accept_id']].count().sort_values(['accept_id'], ascending=False)
#保留上述groupby后前38行数据，其余置为默认值，两条语句共用可用于归并数据
train.loc[train[-train['accept_org_id'].isin(df.index[0:39].tolist())].index, 'accept_org_id'] = '-999' 
#統計某列的佔比
group = train.groupby(['PRICE_ID']).agg({'IS_SUCC': ['sum','count']})
group.columns = [ 'SUM','COUNT' ]
group['PRICE_ID_RATE']=group['SUM']/group['COUNT']
group.reset_index(inplace=True)
group=group[['PRICE_ID','PRICE_ID_RATE']]
df = pd.merge(df, group[['PRICE_ID','PRICE_ID_RATE']], on=['PRICE_ID'], how='left')
#衍生字段
df['BRD_CNT_AVG'] = df.loc[:,['BRD_CNT_11', 'BRD_CNT_10', 'BRD_CNT_09']].mean(axis=1)
df['BRD_CNT_FLU'] = (df.loc[:,['BRD_CNT_11', 'BRD_CNT_10', 'BRD_CNT_09']].max(axis=1) - df.loc[:,['BRD_CNT_11', 'BRD_CNT_10', 'BRD_CNT_09']].min(axis=1))/df['BRD_CNT_AVG']
all_features['BsmtFinSF1'] = all_features['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
#操作csv
train = pd.read_csv('train_le.csv', dtype=str, encoding="utf-8")
data[['accept_id','complete_code','alarm_name','eqpt_desc','fault_desc',
       'complete_code_name']].to_csv('train_fc.csv', index=False, encoding="utf-8")
#透视表
data.pivot_table(values=["LoanAmount"], index=["Gender","Married","Self_Employed"], aggfunc=np.mean)
#交叉表
pd.crosstab(data["Credit_History"], data["Loan_Status"], margins=True)

#分箱
#qcut可以根据样本分位数对数据进行面元划分，使用的是样本分位数，因此可以得大小基本相等的面元
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
#当然可以设置自定义的分位数（0到1的值）
print(pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.]))
pd.qcut(range(5), 3, labels=["good","medium","bad"])
#或者手工指定
bins = [0, 7, 12, 14, 18, 21, 24]
group_name = ['catg0', 'catg1', 'catg2', 'catg3', 'catg4', 'catg5']
train['accept_hour'] = pd.cut(train['accept_hour'], bins, labels=group_name)
# 查看切分后的属性与target属性Survive的关系
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
#用固定宽度的箱进行量化计数
### Generate 20 random integers uniformly between 0 and 99
small_counts = np.random.randint(0, 100, 20)
np.floor_divide(small_counts, 10)
### An array of counts that span several magnitudes
large_counts = [
    296, 8286, 64011, 80, 3, 725, 867, 2215, 7689, 11495, 91897, 44, 28, 7971,
    926, 122, 22222
]
np.floor(np.log10(large_counts))
#任意变量的分箱
#Binning:
def binning(col, cut_points, labels=None):
    #Define min and max values:
    minval = col.min()
    maxval = col.max()   
    #create list by adding min and max to cut_points
    break_points = [minval] + cut_points + [maxval]
    #if no labels provided, use default labels 0 ... (n-1)
    if not labels:
        labels = range(len(cut_points)+1)
    #Binning using cut function of pandas
    colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
    return colBin
#Binning age:
cut_points = [90,140,190]
labels = ["low","medium","high","very high"]
data["LoanAmount_Bin"] = binning(data["LoanAmount"], cut_points, labels)
print(pd.value_counts(data["LoanAmount_Bin"], sort=False))
#年龄分箱
data['age_seg'] = pd.cut(data['age'], [-100, 7, 14, 20, 27, 35, 40, 60, 120], labels=range(8))

#为名义变量编码
def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded
#Coding LoanStatus as Y=1, N=0:
print(pd.value_counts(data["Loan_Status"]))
data["Loan_Status_Coded"] = coding(data["Loan_Status"], {'N':0,'Y':1})
print(pd.value_counts(data["Loan_Status_Coded"]))

# 建立object属性映射字典  
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royalty":5, "Officer": 6}
dataset['Title'] = dataset['Title'].map(title_mapping)

#类型转化
for i, row in colTypes.iterrows():  #i: dataframe index; row: each row in series format
    if row['type']=="categorical":
        data[row['feature']]=data[row['feature']].astype(np.object)
    elif row['type']=="continuous":
        data[row['feature']]=data[row['feature']].astype(np.float)
print(data.dtypes)

#时间窗口
actions = None
for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
    start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
    start_days = start_days.strftime('%Y-%m-%d')
    if actions is None:
        actions = get_action_feat(start_days, train_end_date)
    else:
        actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',on=['user_id', 'sku_id'])

#log对数变换
#对数变换是处理具有重尾分布的正数的有力工具。（重尾分布在尾部范围内的概率比高斯分布的概率大）。它将分布在高端的长尾压缩成较短的尾部，并将低端扩展成较长的头部。
biz_df['log_review_count'] = np.log(biz_df['review_count'] + 1)
for col in big_num_cols:
    train[col] = train[col].map(lambda x: np.log1p(x))

#特征交叉
#这一点对基于决策树的模型没有影响，但发交互特征对广义线性模型通常很有帮助，主要针对数值型特征
X2 = preproc.PolynomialFeatures(include_bias=False).fit_transform(X)
#计算统计特征
def featureCount(data):
    def feature_count(data, features=[]):
        new_feature = 'count'
        for i in features:
            new_feature += '_' + i
        temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})
        data = data.merge(temp, 'left', on=features)
        return data
    data = feature_count(data, ['communityName'])
    data = feature_count(data, ['buildYear'])
    data = feature_count(data, ['totalFloor'])
    data = feature_count(data, ['communityName', 'totalFloor'])
    data = feature_count(data, ['communityName', 'newWorkers'])
    data = feature_count(data, ['communityName', 'totalTradeMoney'])
    return data    
train = featureCount(train)

#特征选择
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris = load_iris()
X, y = iris.data, iris.target
X.shape
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
clf.feature_importances_
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape
# 基于方差筛选  如果先做标准化，方差算出来结果全为1 
from sklearn.feature_selection import VarianceThreshold
threshhold_var = 0.3   # 事先设置阈值
Arr_X_Cols_Var  = VarianceThreshold().fit(X).variances_        # ndarray，每个变量的方差
idx_Cols_Var_d  = [ i for i in range(len(Arr_X_Cols_Var)) if Arr_X_Cols_Var[i] < threshhold_var]
Cols_Var_Filter = X.iloc[:,idx_Cols_Var_d].columns    # 这些字段名要过滤掉
print('待删除字段为：', Cols_Var_Filter)
X = X.drop(Cols_Var_Filter, axis=1)
X.head()
#卡方選擇
from sklearn.feature_selection import SelectKBest
SelectKBest(chi2, k=2).fit_transform(X, y)

#聚類
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
kmeans=KMeans(n_clusters=4).fit(df1)
kmeans.labels_
kmeans.cluster_centers_
pred=kmeans.predict(df1)
df['KM_RESULT']=pd.DataFrame(pred)
silhouette_score(data_x,kmeans.labels_,metric='euclidean') # 计算轮廓系数耗时很久
#高斯混合聚类
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4)
gmm.fit(data)
labels = gmm.predict(data)
frame = pd.DataFrame(data)
frame['cluster'] = labels
frame.columns = ['Weight', 'Height', 'cluster']
color=['blue','green','cyan', 'black']
for k in range(0,4):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["Weight"],data["Height"],c=color[k])
plt.show()

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
data['cluster']= pd.DataFrame(gmm.fit_predict(data[col]))
# 聚类后构造统计特征
for feature1 in col1:
    for feature2 in col2:
        temp = data.groupby(['cluster',feature1])[feature2].agg('mean').reset_index(name=feature2+'_'+feature1+'_cluster_mean')
        temp.fillna(0, inplace=True)       
        data = data.merge(temp, on=['cluster', feature1], how='left')

#采样
DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
#n是要抽取的行数。
#frac是抽取的比列。
#replace抽样后的数据是否代替原DataFrame()
#weights这个是每个样本的权重
#axis是选择抽取数据的行还是列。axis=0的时是抽取行，axis=1时是抽取列
train_sam = pd.concat([X_train[X_train['class'] == 0].sample(n=10000,random_state=42,replace=False), 
                       X_train[X_train['class'] == 1].sample(n=10000,random_state=42,replace=False),
                       X_train[X_train['class'] == 2].sample(n=10000,random_state=42,replace=False),
                       X_train[X_train['class'] == 3],
                       X_train[X_train['class'] == 4],
                       X_train[X_train['class'] == 6],
                       X_train[X_train['class'] == 7].sample(n=10000,random_state=42,replace=False),
                       X_train[X_train['class'] == 8],
                       X_train[X_train['class'] == 9]
                      ], ignore_index=True)
#取前几个类别来采样
trainSam = pd.DataFrame(columns=train.columns)
for var in df.index[0:400].tolist():
    tmp = train[train['complete_code'] == var].sample(frac=0.1,replace=False,random_state=0,axis=0)
    trainSam = pd.concat([trainSam,tmp],axis=0)

# 按某字段取值分层抽样
from sklearn.utils import shuffle
group = train.groupby(['KM_RESULT']).agg({'IS_SUCC': ['sum','count']})
group.columns = [ 'SUM','COUNT' ]
group['KM_RESULT_SORT']=group['COUNT']/len(train.loc[train['IS_SUCC']==0])
succ_rate=len(train.loc[train['IS_SUCC']==1])/len(train.loc[train['IS_SUCC']==0])
aa=list(group.index)
sample=train.loc[train['IS_SUCC']==-1]
data0 = train.loc[(train["IS_SUCC"] == 0)]
for i in range(0,len(aa)):
    print(aa[i])
    group.loc[aa[i]].KM_RESULT_SORT
    data3 = data0.loc[(data0["KM_RESULT"] == aa[i])].sample(frac=group.loc[aa[i]].KM_RESULT_SORT*succ_rate*1.2) #.iloc[:10000, :]
    sample=shuffle(pd.concat([sample,data3],ignore_index=True))#.drop_duplicates()

#多分类输出
preds = lr2.predict_proba(test[features])
preds = np.argmax(preds, axis=1)
test['pre_alarm'] = pd.DataFrame(preds)

#--------------------------可视化--------------------------
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#柱状图
data.boxplot(column="ApplicantIncome",by="Loan_Status")
data.hist(column="ApplicantIncome",by="Loan_Status",bins=30)
#风琴图
data['dur'] = data['dur'].apply(lambda x: 5 if x>=5 else x)
sns.boxplot(x="class", y="dur", data=data)
sns.violinplot(x="class", y="dur", data=data, palette="muted")
#热力图
conmat = metrics.confusion_matrix(test['alarm'], test['pre_alarm'])
fig, ax = plt.subplots(figsize = (15, 15))
sns.heatmap(conmat, cmap='YlGnBu', annot=True, fmt="d", ax = ax)
#dataframe的箱線圖
fig,axes = plt.subplots(1,8,figsize=(15,7))
color = dict(boxes='DarkGreen', whiskers='DarkOrange',
              medians='DarkBlue', caps='Red')
# boxes表示箱体，whisker表示触须线
# medians表示中位数，caps表示最大与最小值界限
tips=df[[ 'PREFER_VALUE', 'TM_CHARGE', 'ARPU_VALUE', 'CALL_AMOUNT', 'CALLED_AMOUNT',
 'SJ_GROUP_VOLUME',  'JL_CALLING_LIMIT_DUR', 'JL_CALLING_DUR']] 
tips.plot(kind='box',ax=axes,subplots=True,
                              title='Different boxplots',color=color,sym='r+')

fig.subplots_adjust(wspace=4,hspace=2)  # 调整子图之间的间距

# roc/auc 曲线
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds_keras = roc_curve(Y_valid, Y_pred)
auc = auc(fpr, tpr)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

#------------------类别字段处理----------------
#onehot编码
train = pd.get_dummies(train, columns=['accept_org_id','serv_item','sub_serv_type'])
# 基数比较高的
# count 编码
communityName_cnts = data['communityName'].value_counts().reset_index()
# 计数编码
df.groupby(['category'])['target'].transform(sum)
# WOE编码
from category_encoders import *
import pandas as pd
from sklearn.datasets import load_boston
bunch = load_boston()
y = bunch.target > 22.5
X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
enc = WOEEncoder(cols=['CHAS', 'RAD']).fit(X, y)
numeric_dataset = enc.transform(X)
# target encoding 目标编码
from category_encoders import *
import pandas as pd
from sklearn.datasets import load_boston
bunch = load_boston()
y = bunch.target
X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
enc = TargetEncoder(cols=['CHAS', 'RAD']).fit(X, y)
numeric_dataset = enc.transform(X)
print(numeric_dataset.info())
# 采用交叉验证获得上述编码
def mean_woe_target_encoder(train,test,target,col,n_splits=10):
    folds = StratifiedKFold(n_splits)

    y_oof = np.zeros(train.shape[0])
    y_oof_2= np.zeros(train.shape[0])
    y_test_oof = np.zeros(test.shape[0]).reshape(-1,1)
    y_test_oof2 = np.zeros(test.shape[0]).reshape(-1,1)

    splits = folds.split(train, target)
    
    for fold_n, (train_index, valid_index) in enumerate(splits):
        X_train, X_valid = train[col].iloc[train_index], train[col].iloc[valid_index]
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        clf=ce.target_encoder.TargetEncoder()
    
    #    dtrain = lgb.Dataset(X_train, label=y_train)
    #    dvalid = lgb.Dataset(X_valid, label=y_valid)
    
        #clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=1, early_stopping_rounds=500)
        clf.fit(X_train.values,y_train.values)    
        y_pred_valid = clf.transform(X_valid.values)
        y_oof[valid_index] = y_pred_valid.values.reshape(1,-1)

        tp=(clf.transform(test[col].values)/(n_splits*1.0)).values
        tp=tp.reshape(-1,1)
        y_test_oof+=tp    
    
        del X_train, X_valid, y_train, y_valid
        gc.collect()    
        
    for fold_n, (train_index, valid_index) in enumerate(splits):
        X_train, X_valid = train[col].iloc[train_index], train[col].iloc[valid_index]
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        clf=ce.woe.WOEEncoder()
    
    #    dtrain = lgb.Dataset(X_train, label=y_train)
    #    dvalid = lgb.Dataset(X_valid, label=y_valid)
    
        #clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=1, early_stopping_rounds=500)
        clf.fit(X_train.values,y_train.values)    
        y_pred_valid = clf.transform(X_valid.values)
        y_oof2[valid_index] = y_pred_valid.values.reshape(1,-1)
    
        tp=(clf.transform(test[col].values)/(n_splits*1.0)).values
        tp=tp.reshape(-1,1)
        y_test_oof2+=tp    
        del X_train, X_valid, y_train, y_valid
        gc.collect()     
    return y_oof,y_oof_2,y_test_oof,y_test_oof2

# mean encoding 均值编码
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold
from itertools import product
 
class MeanEncoder:
    def __init__(self, categorical_features, n_splits=5, target_type='classification', prior_weight_func=None):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode
 
        :param n_splits: the number of splits used in mean encoding
 
        :param target_type: str, 'regression' or 'classification'
 
        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        """
 
        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}
 
        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None
 
        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

                    
    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()
 
        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()
 
        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)
 
        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values
 
        return nf_train, nf_test, prior, col_avg_y
 
    def fit_transform(self, X, y): 
        
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)
 
        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new
 
    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
 
        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
 
        return X_new