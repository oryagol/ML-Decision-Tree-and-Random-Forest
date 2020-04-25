import numpy as np
import pandas as pd
#import Section_B as b
import random
import sklearn.metrics as met

data = pd.read_excel('adult_income.xlsx') #the whole data from the excel file
feature_desc = pd.read_excel('adult_income.xlsx', 'feature_desc') #the other tab that describes the features
feature_names = data.columns #feature names
feature_names = feature_names.tolist()
for name in feature_names:
    name= name[0:len(name)-1]
feature_names.remove('>50K')
feature_names.remove('fnlwgt')
train_data = data[:19035] #training data set
validation_data = data[19035:24131] #validation data set
test_data = data[24131:] #test data set
feature_vals = {} #dictionary for feature and its instances
feature_numeric_vals = {} #dictionary for numeric feature and its instances
feature_regression = "education-num" #the name of the feature of the value we want to predict
forest = [] # a list to save all the trees of the random forest
multy_target = {} # a dictionary for all of the values of the feature that we want to predict in multyclassification
feature_multyclassification="education"
#create the dictionaries divided to numeric and non numeric
def featVal(mode):
    if mode:
        for i in range(feature_desc.shape[0]-2):
            row = feature_desc.iloc[i]
            feat_name = row[0]
            feat_name = feat_name[0:len(row[0])-1]
            if row[1] != 'continuous.':
                feature_vals[feat_name] = set(row[1:])
                feature_vals[feat_name] = [x for x in feature_vals[feat_name] if str(x) != 'nan']
                sorted(feature_vals[feat_name])
            else:
                feat = feat_name[0:len(row[0])-1]
                if feat != "fnlwgt":
                    feature_numeric_vals[feat] = sorted(list(set(data[feat])))
    else:
        features_list = ["age", "workclass", "marital-status", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
        for i in range(feature_desc.shape[0]-2):
            row = feature_desc.iloc[i]
            feat_name = row[0]
            feat_name = feat_name[0:len(row[0])-1]
            if feat_name in features_list:
                if row[1] != 'continuous.':
                    feature_vals[feat_name] = set(row[1:])
                    feature_vals[feat_name] = [x for x in feature_vals[feat_name] if str(x) != 'nan']
                    sorted(feature_vals[feat_name])
                else:
                    feat = feat_name[0:len(row[0])-1]
                    feature_numeric_vals[feat] = sorted(list(set(data[feat])))
            elif feat_name == feature_multyclassification:
                multy_target[feat_name] = set(row[1:])
                multy_target[feat_name] = [x for x in multy_target[feat_name] if str(x) != 'nan']
                multy_target[feat_name] = sorted(multy_target[feat_name])

#checks for each feature the count of his instances
def feature_count(curr_data, predict):
    countOfInstance = {}
    for feature in feature_vals:
        countOfInstance[feature] = curr_data.groupby([feature, predict]).count()
        countOfInstance[feature] = countOfInstance[feature].iloc[:,1]
    return countOfInstance

# split the data by a non numeric feature
def split_data(curr_data, feature, instance):
    right = curr_data.loc[curr_data[feature] != instance]
    left = curr_data.loc[curr_data[feature] == instance]
    return left, right

# split the data by a numeric feature
def split_numeric_data(curr_data, feature, threshold):
    right = curr_data.loc[curr_data[feature] < threshold]
    left = curr_data.loc[curr_data[feature] >= threshold]
    return left, right


# calculate the gini of every feature to decide the node question
def calc_gini_feature(curr_data, feature_vals, feature_numeric_vals, predict):
    min_gini = 1
    if len(curr_data) == 0:
        return min_gini, min_gini, min_gini, None, None
    for feature in feature_vals.keys():
        for instance in feature_vals[feature]:
            left, right = split_data(curr_data, feature, instance)
            left_impurity = calc_gini_instance(left,  predict)
            right_impurity = calc_gini_instance(right, predict)
            avg = (len(left)/len(curr_data))*left_impurity+(len(right)/len(curr_data))*right_impurity
            if min_gini > avg:
                min_gini = avg
                best_feature = feature
                best_instance = instance
                best_left_impurity = left_impurity
                best_right_impurity = right_impurity
    numeric_min_gini = 1
    for feature in feature_numeric_vals.keys():
        for number in feature_numeric_vals[feature]:
            left, right = split_numeric_data(curr_data, feature, number)
            left_impurity = calc_gini_instance(left,  predict)
            right_impurity = calc_gini_instance(right, predict)
            avg = (len(left)/len(curr_data))*left_impurity+(len(right)/len(curr_data))*right_impurity
            if numeric_min_gini > avg:
                numeric_min_gini = avg
                best_numeric_feature = feature
                best_numeric_instance = number
                best_left_numeric_impurity = left_impurity
                best_right_numeric_impurity = right_impurity
    if min_gini == 1 and numeric_min_gini == 1:
        return 1, None, None, None, None
    if min_gini > numeric_min_gini:
        return numeric_min_gini, best_left_numeric_impurity, best_right_numeric_impurity, best_numeric_feature, best_numeric_instance
    else:
        return min_gini,best_left_impurity,best_right_impurity ,best_feature, best_instance

# calculate the mse to decide the question
def calc_mse(curr_data, feature_name):
    if len(curr_data) == 0:
        return -1
    avg = np.mean(curr_data[feature_name])
    education_num = curr_data[feature_name]
    education_num = education_num.tolist()
    sum=0
    for i in range(len(education_num)):
        sum += (education_num[i]-avg)**2
    return sum/len(education_num)

# calculate the mse of every feature to decide the node question
def calc_mse_feature(curr_data,feature_name, feature_vals, feature_numeric_vals):
    isFirst = True
    best_mse = 0.0
    for feature in feature_vals.keys():
        for instance in feature_vals[feature]:
            left, right = split_data(curr_data, feature, instance)
            left_impurity = calc_mse(left, feature_name)
            if left_impurity == -1:
                continue
            right_impurity = calc_mse(right, feature_name)
            if right_impurity == -1:
                continue
            avg = (len(left)/len(curr_data))*left_impurity+(len(right)/len(curr_data))*right_impurity
            if isFirst:
                isFirst = False
                best_mse = avg
                best_feature = feature
                best_instance = instance
                best_left_impurity = left_impurity
                best_right_impurity = right_impurity
            else:
                if avg < best_mse:
                    best_mse = avg
                    best_feature = feature
                    best_instance = instance
                    best_left_impurity = left_impurity
                    best_right_impurity = right_impurity
    isFirst = True
    best_numeric_mse = 0.0
    for feature in feature_numeric_vals.keys():
        for instance in feature_numeric_vals[feature]:
            left, right = split_numeric_data(curr_data, feature, instance)
            left_impurity = calc_mse(left, feature_name)
            if left_impurity == -1:
                continue
            right_impurity = calc_mse(right, feature_name)
            if right_impurity == -1:
                continue
            avg = (len(left) / len(curr_data)) * left_impurity + (len(right) / len(curr_data)) * right_impurity
            if isFirst:
                isFirst = False
                best_numeric_mse = avg
                best_numeric_feature = feature
                best_numeric_instance = instance
                best_left_numeric_impurity = left_impurity
                best_right_numeric_impurity = right_impurity
            else:
                if avg < best_numeric_mse:
                    best_numeric_mse = avg
                    best_numeric_feature = feature
                    best_numeric_instance = instance
                    best_left_numeric_impurity = left_impurity
                    best_right_numeric_impurity = right_impurity
    if best_mse == 0.0 and best_numeric_mse == 0.0:
        return 0, 0, 0, None, None
    elif best_mse == 0.0:
        return best_numeric_mse,best_left_numeric_impurity,best_right_numeric_impurity,best_numeric_feature,best_numeric_instance
    elif best_numeric_mse == 0.0:
        return best_mse, best_left_impurity, best_right_impurity, best_feature, best_instance
    else:
        if best_mse < best_numeric_mse:
            return  best_mse, best_left_impurity, best_right_impurity, best_feature, best_instance
        else:
            return best_numeric_mse,best_left_numeric_impurity,best_right_numeric_impurity,best_numeric_feature,best_numeric_instance



# calculate the gini an instance of a feature to decide the question
def calc_gini_instance(curr_data, predict):
    update_list = []
    curr_vals = set(curr_data[predict].tolist())
    for val in curr_vals:
        if val in curr_vals:
            update_list.append(val)
    value_list = sorted(update_list)
    grouped = curr_data.groupby([predict]).count()
    grouped = grouped.iloc[:, 1]
    valsCount = {}
    sum = 0
    if len(grouped) == 1:
        return 0.0
    for i in range(len(value_list)):
        valsCount[grouped[i]] = value_list[i]
        sum += grouped[i]
    impurity = 1
    if sum == 0:
        return impurity
    for val in grouped:
        impurity -= (val / sum) ** 2
    return impurity



# a function that randomly choose features based on a number that you send
def choose_features(num, feature_list):
    chosenFeatures = []
    i = 0
    if 'fnlwgt' in feature_list:
        feature_list.remove('fnlwgt')
    while i < num:
        randNum = random.randint(0,len(feature_list)-1)
        if chosenFeatures.__contains__(feature_list[randNum]):
            continue
        else:
            chosenFeatures.append(feature_list[randNum])
            i += 1
    numeric_chosen = {}
    non_numeric_chosen = {}
    for feature in chosenFeatures:
        if feature in feature_vals:
            non_numeric_chosen[feature] = feature_vals[feature]
        elif feature in feature_numeric_vals:
            numeric_chosen[feature] = feature_numeric_vals[feature]
    return non_numeric_chosen, numeric_chosen


# a class of a decision tree
class decisionTree:
    def __init__(self, imp, feature , instance, left, right, depth, max_depth, data):
        self.imp = imp
        self.feature = feature
        self.instance = instance
        self.right = right
        self.left = left
        self.depth = depth
        self.max_depth = max_depth
        self.data = data

    def __str__(self):
        return "feature is "+str(self.feature)+" instance is "+str(self.instance)+" depth is "+str(self.depth) + " and impurrity is "+str(self.imp)

    def is_numerical(self,numerical_features,categorial_features):
        if self.feature in numerical_features.keys():
            return True
        elif self.feature in categorial_features.keys():
            return False
        else:
            return None

    def multyClassify(self, predict):
        update_list = []
        curr_vals = set(self.data[predict].tolist())
        for val in curr_vals:
            if val in curr_vals:
                update_list.append(val)
        value_list = sorted(update_list)
        grouped = self.data.groupby([predict]).count()
        grouped = grouped.iloc[:, 1]
        valsCount = {}
        for i in range(len(value_list)):
            valsCount[grouped[i]] = value_list[i]
        max = np.max(grouped)
        return valsCount[max]

    def classify(self, predict):
        update_list = []
        curr_vals = set(self.data[predict].tolist())
        for val in curr_vals:
            if val in curr_vals:
                update_list.append(val)
        value_list = sorted(update_list)
        grouped = self.data.groupby([predict]).count()
        grouped = grouped.iloc[:, 1]
        if len(grouped) == 1:
            return list(set(self.data[predict]))[0]
        valsCount = {}
        for i in range(len(value_list)):
            valsCount[grouped[i]] = value_list[i]
        max = np.max(grouped)
        return valsCount[max]

# a function to create a decision tree of classification
def create_classification_tree(train_data, max_depth, isForest, feature_vals, feature_numeric_vals, num_of_features, predict):
    if isForest:
        feature_vals ,feature_numeric_vals = choose_features(num_of_features, feature_names)
    imp, imp_left, imp_right, feature, instance = calc_gini_feature(train_data,  feature_vals ,feature_numeric_vals, predict)
    root = decisionTree(imp=imp, feature=feature, instance=instance,left=None,right=None, depth=0, max_depth=max_depth, data=train_data)
    return root
# a function to create a decision tree of regression
def create_regression_tree(train_data, max_depth, isForest, feature_vals, feature_numeric_vals, num_of_features):
    if isForest:
        feature_vals ,feature_numeric_vals = choose_features(num_of_features, feature_names)
        imp, imp_left, imp_right, feature, instance = calc_mse_feature(train_data,feature_regression,  feature_vals ,feature_numeric_vals)
    else:
        imp, imp_left, imp_right, feature, instance = calc_mse_feature(train_data,feature_regression,  feature_vals ,feature_numeric_vals)
    root = decisionTree(imp=imp, feature=feature, instance=instance,left=None,right=None, depth=0, max_depth=max_depth, data=train_data)
    return root

# recursive function to expand the tree until the max depth
def expand_classification_tree(anc, min_factor, isForest, feature_vals, feature_numeric_vals, num_of_features, predict):
    if anc.depth == 0:
        if anc.feature in feature_vals:
            catagorial = True
        else:
            catagorial = False
    if isForest:
        feature_vals ,feature_numeric_vals = choose_features(num_of_features, feature_names)
    if anc.depth != 0:
        imp, imp_left, imp_right, feature, instance = calc_gini_feature(anc.data,feature_vals, feature_numeric_vals, predict)
        if feature in feature_vals:
            catagorial = True
        else:
            catagorial = False
        if imp == 1:
            return
        anc.feature = feature
        anc.instance = instance
        if anc.imp == 0.0:
            return
    else:
        imp_left = anc.imp
        imp_right = anc.imp
    if anc.depth == anc.max_depth:
        return
    if catagorial:
        left, right = split_data(anc.data, anc.feature, anc.instance)
    else:
        left, right = split_numeric_data(anc.data, anc.feature, anc.instance)
    if len(left) < min_factor or len(right) < min_factor:
        return
    anc.left = decisionTree(imp=imp_left, feature=None, instance=None, left=None, right=None, depth=anc.depth+1, max_depth=anc.max_depth, data=left)
    anc.right = decisionTree(imp=imp_right, feature=None, instance=None,left=None, right=None, depth=anc.depth+1, max_depth=anc.max_depth, data=right)
    expand_classification_tree(anc.left, min_factor, isForest, feature_vals, feature_numeric_vals, num_of_features, predict)
    expand_classification_tree(anc.right, min_factor, isForest, feature_vals, feature_numeric_vals, num_of_features, predict)
    return anc

# recursive function to expand the tree until the max depth
def expand_regression_tree(anc, min_factor, isForest, feature_vals, feature_numeric_vals, num_of_features):
    if anc.depth != 0:
        if isForest:
            feature_vals, feature_numeric_vals = choose_features(num_of_features, feature_names)
        imp, imp_left, imp_right, feature, instance = calc_mse_feature(anc.data,feature_regression,feature_vals, feature_numeric_vals)
        if feature == None:
            return
        anc.feature = feature
        anc.instance = instance
    if anc.imp == 0.0:
        return
    else:
        imp_left = anc.imp
        imp_right = anc.imp
        feature = anc.feature
    if anc.depth == anc.max_depth:
        return
    if feature in feature_vals:
        left, right = split_data(anc.data, anc.feature, anc.instance)
    elif feature in feature_numeric_vals:
        left, right = split_numeric_data(anc.data, anc.feature, anc.instance)
    if len(left) < min_factor or len(right) < min_factor:
        return
    anc.left = decisionTree(imp=imp_left, feature=None, instance=None, left=None, right=None, depth=anc.depth+1, max_depth=anc.max_depth, data=left)
    anc.right = decisionTree(imp=imp_right, feature=None, instance=None,left=None, right=None, depth=anc.depth+1, max_depth=anc.max_depth, data=right)
    expand_regression_tree(anc.left, min_factor, isForest, feature_vals, feature_numeric_vals, num_of_features)
    expand_regression_tree(anc.right, min_factor, isForest, feature_vals, feature_numeric_vals, num_of_features)
    return anc
# a function to print the tree
def p_tree(anc):
    if anc.left is None and anc.right is None:
        return
    print(anc)
    p_tree(anc.left)
    p_tree(anc.right)

# a function to get random data for a tree in the random forest
def prepare_data(train_data, num):
    return train_data.sample(int(len(train_data)/num))

# a function to build a random forest of classification
def build_class_forest(forest_size, split_data_factor,min_factor,max_depth, feature_vals, feature_numeric_vals, num_of_features, predict):
    for i in range(forest_size):
        root = create_classification_tree(prepare_data(train_data, split_data_factor),max_depth, True, feature_vals, feature_numeric_vals, num_of_features, predict)
        tree = expand_classification_tree(root, min_factor, True, feature_vals, feature_numeric_vals, num_of_features, predict)
        forest.append(tree)
        print("finished " + "tree "+ str(i+1))
    return forest
# a function to build a random forest of regression
def build_reg_forest(forest_size, split_data_factor,min_factor,max_depth, feature_vals, feature_numeric_vals, num_of_features):
    for i in range(forest_size):
        root = create_regression_tree(prepare_data(train_data, split_data_factor),max_depth, True, feature_vals, feature_numeric_vals, num_of_features)
        tree = expand_regression_tree(root, min_factor, True, feature_vals, feature_numeric_vals, num_of_features)
        forest.append(tree)
        print("finished " + "tree "+ str(i+1))
    return forest

# calculate the gini an instance of a feature to decide the question
def calc_gini_instance_multy(curr_data, predict):
        update_list = []
        curr_vals = set(curr_data[predict].tolist())
        for val in curr_vals:
            if val in curr_vals:
                update_list.append(val)
        value_list = sorted(update_list)
        grouped = curr_data.groupby([predict]).count()
        grouped = grouped.iloc[:, 1]
        valsCount = {}
        sum = 0
        for i in range(len(value_list)):
            valsCount[grouped[i]] = value_list[i]
            sum += grouped[i]
        impurity = 1
        if sum == 0:
            return impurity
        for val in grouped:
            impurity -= (val / sum) ** 2
        return impurity


# calculate the gini of every feature to decide the node question
def calc_gini_feature_multy(curr_data, feature_vals, feature_numeric_vals, predict):
    min_gini = 1
    if len(curr_data) == 0:
        return min_gini, min_gini, min_gini, None, None
    for feature in feature_vals.keys():
        for instance in feature_vals[feature]:
            left, right = split_data(curr_data, feature, instance)
            left_impurity = calc_gini_instance_multy(left, predict)
            right_impurity = calc_gini_instance_multy(right, predict)
            avg = (len(left)/len(curr_data))*left_impurity+(len(right)/len(curr_data))*right_impurity
            if min_gini > avg:
                min_gini = avg
                best_feature = feature
                best_instance = instance
                best_left_impurity = left_impurity
                best_right_impurity = right_impurity
    numeric_min_gini = 1
    for feature in feature_numeric_vals.keys():
        for number in feature_numeric_vals[feature]:
            left, right = split_numeric_data(curr_data, feature, number)
            left_impurity = calc_gini_instance_multy(left, predict)
            right_impurity = calc_gini_instance_multy(right, predict)
            avg = (len(left)/len(curr_data))*left_impurity+(len(right)/len(curr_data))*right_impurity
            if numeric_min_gini > avg:
                numeric_min_gini = avg
                best_numeric_feature = feature
                best_numeric_instance = number
                best_left_numeric_impurity = left_impurity
                best_right_numeric_impurity = right_impurity
    if min_gini == 1 and numeric_min_gini == 1:
        return 1,None, None, None, None
    if min_gini > numeric_min_gini:
        return numeric_min_gini, best_left_numeric_impurity, best_right_numeric_impurity, best_numeric_feature, best_numeric_instance
    else:
        return min_gini,best_left_impurity,best_right_impurity ,best_feature, best_instance
# a function to create a decision tree of multyclassification
def create_multy_classification_tree(train_data, max_depth, isForest, feature_vals, feature_numeric_vals, num_of_features, predict):
    if isForest:
        feature_vals ,feature_numeric_vals = choose_features(num_of_features, feature_names)
        imp, imp_left, imp_right, feature, instance = calc_gini_feature_multy(train_data,  feature_vals ,feature_numeric_vals, predict)
    else:
        imp, imp_left, imp_right, feature, instance = calc_gini_feature_multy(train_data,  feature_vals ,feature_numeric_vals, predict)
    root = decisionTree(imp=imp, feature=feature, instance=instance,left=None,right=None, depth=0, max_depth=max_depth, data=train_data)
    return root

# recursive function to expand the tree until the max depth
def expand_multy_classification_tree(anc, min_factor, isForest, feature_vals, feature_numeric_vals, num_of_features, predict):
    if anc.depth == 0:
        if anc.feature in feature_vals:
            catagorial = True
        else:
            catagorial = False
    if isForest:
        feature_vals, feature_numeric_vals = choose_features(num_of_features, feature_names)
    if anc.depth != 0:
        imp, imp_left, imp_right, feature, instance = calc_gini_feature(anc.data, feature_vals, feature_numeric_vals,
                                                                        predict)
        if feature in feature_vals:
            catagorial = True
        else:
            catagorial = False
        if imp == 1:
            return
        anc.feature = feature
        anc.instance = instance
        if anc.imp == 0.0:
            return
    else:
        imp_left = anc.imp
        imp_right = anc.imp
    if anc.depth == anc.max_depth:
        return
    if catagorial:
        left, right = split_data(anc.data, anc.feature, anc.instance)
    else:
        left, right = split_numeric_data(anc.data, anc.feature, anc.instance)
    if len(left) < min_factor or len(right) < min_factor:
        return
    anc.left = decisionTree(imp=imp_left, feature=None, instance=None, left=None, right=None, depth=anc.depth + 1,
                            max_depth=anc.max_depth, data=left)
    anc.right = decisionTree(imp=imp_right, feature=None, instance=None, left=None, right=None, depth=anc.depth + 1,
                             max_depth=anc.max_depth, data=right)
    expand_multy_classification_tree(anc.left, min_factor, isForest, feature_vals, feature_numeric_vals, num_of_features,
                               predict)
    expand_multy_classification_tree(anc.right, min_factor, isForest, feature_vals, feature_numeric_vals, num_of_features,
                               predict)
    return anc
# a function to build a random forest of multyclassification
def build_multyclass_forest(forest_size, split_data_factor,min_factor,max_depth, feature_vals, feature_numeric_vals, num_of_features, predict):
    for i in range(forest_size):
        root = create_multy_classification_tree(prepare_data(train_data, split_data_factor),max_depth, True, feature_vals, feature_numeric_vals, num_of_features, predict)
        tree = expand_multy_classification_tree(root, min_factor, True, feature_vals, feature_numeric_vals, num_of_features, predict)
        forest.append(tree)
        print("finished " + "tree "+ str(i+1))
    return forest


