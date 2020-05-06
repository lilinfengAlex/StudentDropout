import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=np.inf)

df = pd.read_excel('FINAL With Fake ID_NUM.XLSX')
print(df.head())

# # Add a column to the dataset for Student dropout(0/1)
# GPA = df['DegreeGPA'].values
# # print(GPA)
# label = np.where(np.isnan(GPA), 0, 1)
# # print(label)
label = df['label'].values
print(label)
df = df.drop(['term', 'ID_NUM', 'label'], axis=1)
print(df.head())


# Data Preprocessing
sex_dic = {
    "M": 0,
    "F": 1,
}

plan10 = df['plan10'].values
plan10_dic = {}
for i in plan10:
    if type(i) is float and np.isnan(i):
        continue
    elif i not in plan10_dic.keys():
        plan10_dic[i] = len(plan10_dic) + 1
print(plan10_dic)

subpln10 = df['subpln10'].values
subpln10_dic = {}
for i in subpln10:
    if type(i) is float and np.isnan(i):
        continue
    elif i not in subpln10_dic.keys():
        subpln10_dic[i] = len(subpln10_dic) + 1
print('subpln10:\n', subpln10_dic)

plan20 = df['plan20'].values
plan20_dic = {}
for i in plan20:
    if type(i) is float and np.isnan(i):
        continue
    elif i not in plan20_dic.keys():
        plan20_dic[i] = len(plan20_dic) + 1
print('plan20:\n', plan20_dic)


subpln20 = df['subpln20'].values
subpln20_dic = {}
for i in subpln20:
    if type(i) is float and np.isnan(i):
        continue
    elif i not in subpln20_dic.keys():
        subpln20_dic[i] = len(subpln20_dic) + 1
print('subpln20:\n', subpln20_dic)

county = df['county'].values
county_dic = {}
for i in county:
    if not i:
        continue
    elif i not in county_dic.keys():
        county_dic[i] = len(county_dic) + 1
print(county_dic)

state_perm = df['state_perm'].values
state_perm_dic = {}
for i in state_perm:
    if type(i) is float and np.isnan(i):
        continue
    elif i not in state_perm_dic.keys():
        state_perm_dic[i] = len(state_perm_dic) + 1
print(state_perm_dic)

major_type10 = df['major_type10'].values
major_type10_dic = {}
for i in major_type10:
    if type(i) is float and np.isnan(i):
        continue
    elif i not in major_type10_dic.keys():
        major_type10_dic[i] = len(major_type10_dic) + 1
print(major_type10_dic)
# Save
# np.save('major_type10_dic.npy', major_type10_dic)


major_basic = df['major_basic'].values
major_basic_dic = {}
for i in major_basic:
    if type(i) is float and np.isnan(i):
        continue
    elif i not in major_basic_dic.keys():
        major_basic_dic[i] = len(major_basic_dic) + 1
print(major_basic_dic)


DegreeCompletionTermDescr = df['DegreeCompletionTermDescr'].values
DegreeCompletionTermDescr_dic = {}
for i in DegreeCompletionTermDescr:
    # print(i, type(i))
    if type(i) is float and np.isnan(i):
        continue
    elif i not in DegreeCompletionTermDescr_dic.keys():
        DegreeCompletionTermDescr_dic[i] = len(DegreeCompletionTermDescr_dic) + 1
print('DegreeCompletionTermDescr_dic:\n', DegreeCompletionTermDescr_dic)


DegreeAcadPlan = df['DegreeAcadPlan'].values
DegreeAcadPlan_dic = {}
for i in DegreeAcadPlan:
    if type(i) is float and np.isnan(i):
        continue
    elif i not in DegreeAcadPlan_dic.keys():
        DegreeAcadPlan_dic[i] = len(DegreeAcadPlan_dic) + 1
print('DegreeAcadPlan_dic:\n', DegreeAcadPlan_dic)


DegreeDeptName = df['DegreeDeptName'].values
DegreeDeptName_dic = {}
for i in DegreeDeptName:
    if type(i) is float and np.isnan(i):
        continue
    elif i not in DegreeDeptName_dic.keys():
        DegreeDeptName_dic[i] = len(DegreeDeptName_dic) + 1
print('DegreeDeptName_dic:\n', DegreeDeptName_dic)


DegreeSchoolCollegeName = df['DegreeSchoolCollegeName'].values
DegreeSchoolCollegeName_dic = {}
for i in DegreeSchoolCollegeName:
    if type(i) is float and np.isnan(i):
        continue
    elif i not in DegreeSchoolCollegeName_dic.keys():
        DegreeSchoolCollegeName_dic[i] = len(DegreeSchoolCollegeName_dic) + 1
print('DegreeSchoolCollegeName_dic:\n', DegreeSchoolCollegeName_dic)


DegreeAcadProgramDescr = df['DegreeAcadProgramDescr'].values
DegreeAcadProgramDescr_dic = {}
for i in DegreeAcadProgramDescr:
    if type(i) is float and np.isnan(i):
        continue
    elif i not in DegreeAcadProgramDescr_dic.keys():
        DegreeAcadProgramDescr_dic[i] = len(DegreeAcadProgramDescr_dic) + 1
print('DegreeAcadProgramDescr_dic:\n', DegreeAcadProgramDescr_dic)

DegreeSubPlan = df['DegreeSubPlan'].values
DegreeSubPlan_dic = {}
for i in DegreeSubPlan:
    if type(i) is float and np.isnan(i):
        continue
    elif i not in DegreeSubPlan_dic.keys():
        DegreeSubPlan_dic[i] = len(DegreeSubPlan_dic) + 1
print('DegreeSubPlan_dic:\n', DegreeSubPlan_dic)


AID_1_dic = {
    "N": 0,
    "Y": 1,
}

AID_2_dic = {
    "N": 0,
    "Y": 1,
}

AID_3_dic = {
    "N": 0,
    "Y": 1,
}

intl_flag_dic = {
    "N": 0,
    "Y": 1,
}

# Construct data we need
# data = df[['sex', 'credits_term_total_F13', 'AID_2', 'AID_3', 'major_type10']]
# print(data)

x = df.copy()
x.loc[:, 'sex'].replace(sex_dic.keys(), sex_dic.values(), inplace=True)
x.loc[:, 'plan10'].replace(plan10_dic.keys(), plan10_dic.values(), inplace=True)
x.loc[:, 'subpln10'].replace(subpln10_dic.keys(), subpln10_dic.values(), inplace=True)
x.loc[:, 'plan20'].replace(plan20_dic.keys(), plan20_dic.values(), inplace=True)
x.loc[:, 'subpln20'].replace(subpln20_dic.keys(), subpln20_dic.values(), inplace=True)
x.loc[:, 'AID_1'].replace(AID_1_dic.keys(), AID_1_dic.values(), inplace=True)
x.loc[:, 'AID_2'].replace(AID_2_dic.keys(), AID_2_dic.values(), inplace=True)
x.loc[:, 'AID_3'].replace(AID_3_dic.keys(), AID_3_dic.values(), inplace=True)
x.loc[:, 'intl_flag'].replace(intl_flag_dic.keys(), intl_flag_dic.values(), inplace=True)
x.loc[:, 'county'].replace(county_dic.keys(), county_dic.values(), inplace=True)
x.loc[:, 'state_perm'].replace(state_perm_dic.keys(), state_perm_dic.values(), inplace=True)
x.loc[:, 'major_type10'].replace(major_type10_dic.keys(), major_type10_dic.values(), inplace=True)
x.loc[:, 'major_basic'].replace(major_basic_dic.keys(), major_basic_dic.values(), inplace=True)
x.loc[:, 'DegreeCompletionTermDescr'].replace(DegreeCompletionTermDescr_dic.keys(), DegreeCompletionTermDescr_dic.values(), inplace=True)
x.loc[:, 'DegreeAcadPlan'].replace(DegreeAcadPlan_dic.keys(), DegreeAcadPlan_dic.values(), inplace=True)
x.loc[:, 'DegreeDeptName'].replace(DegreeDeptName_dic.keys(), DegreeDeptName_dic.values(), inplace=True)
x.loc[:, 'DegreeSchoolCollegeName'].replace(DegreeSchoolCollegeName_dic.keys(), DegreeSchoolCollegeName_dic.values(), inplace=True)
x.loc[:, 'DegreeAcadProgramDescr'].replace(DegreeAcadProgramDescr_dic.keys(), DegreeAcadProgramDescr_dic.values(), inplace=True)
x.loc[:, 'DegreeSubPlan'].replace(DegreeSubPlan_dic.keys(), DegreeSubPlan_dic.values(), inplace=True)

# fill Nan
x = x.fillna(0)
print(x)

train = x
df = pd.get_dummies(train.iloc[:, 1:-1])
df = df.fillna(df.mean())

X_train = df
y = label


def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=3))
    return (rmse)


# 调用LassoCV函数，并进行交叉验证，默认cv=3
model_lasso = LassoCV(alphas=[0.1, 1, 0.001, 0.0005]).fit(X_train, y)

# 模型所选择的最优正则化参数alpha
print(model_lasso.alpha_)

# 各特征列的参数值或者说权重参数，为0代表该特征被模型剔除了
print(model_lasso.coef_)

# 输出看模型最终选择了几个特征向量，剔除了几个特征向量
coef = pd.Series(model_lasso.coef_, index=X_train.columns)
print(
    "Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")

# 输出所选择的最优正则化参数情况下的残差平均值，因为是3折，所以看平均值
print(rmse_cv(model_lasso).mean())

# 画出特征变量的重要程度，这里面选出前5个重要，后5个不重要的举例
imp_coef = pd.concat([coef.sort_values().head(5),
                      coef.sort_values().tail(5)])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind="barh")
plt.title("Coefficients in the Lasso Model")
plt.show()