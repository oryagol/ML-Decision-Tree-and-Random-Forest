import numpy as np
import pandas as pd
import Section_A as a
import sklearn.metrics as met
import pickle as p
split1 = 19035
split2 = 24131
data = pd.read_excel('adult_income.xlsx') #the whole data from the excel file
train_data = data[:split1] #training data set
validation_data = data[split1:split2] #validation data set
test_data = data[split2:] #test data set

# a method that run classification tree
def classify(tree,row, predict):
    if tree is None:
        return
    if tree.left is None and tree.right is None:
        return tree.classify(predict)
    #categorial
    if tree.is_numerical(a.feature_numeric_vals,a.feature_vals) is False:
         if row[tree.feature]==tree.instance:
            value = classify(tree.left,row, predict)
         else:
            value = classify(tree.right, row, predict)
    #numerical
    else:
        if row[tree.feature] >= tree.instance:
            value = classify(tree.left,row, predict)
        else:
            value = classify(tree.right, row, predict)
    return value

# a method that calculate accuracy of a tree
def calc_accuracy(test_data, tree, start, end, predict):
    count_success = 0
    while start < end:
        row = test_data.loc[start,:]
        true_val = row[predict]
        score = classify(tree, row, predict)
        if true_val == score:
            count_success += 1
        start += 1
    return count_success/len(test_data)

# a method that runs regression tree
def regression(tree, row, predict):
    if tree is None:
        return
    if tree.left is None and tree.right is None:
        return np.mean(tree.data[predict])
    #categorial
    if tree.is_numerical(a.feature_numeric_vals,a.feature_vals) is False:
         if row[tree.feature]==tree.instance:
            value = regression(tree.left,row, predict)
         else:
            value = regression(tree.right, row, predict)
    #numerical
    else:
        if row[tree.feature] >= tree.instance:
            value = regression(tree.left,row, predict)
        else:
            value = regression(tree.right, row, predict)
    return value

# a method that checks the mse of a regression tree
def calc_pred_mse(test_data,tree, start, end, predict):
    predictions=[]
    real_values=[]
    while start < end:
        row = test_data.loc[start, :]
        real_values.append(row[predict])
        predictions.append(regression(tree,row, predict))
        start += 1
    return met.mean_squared_error(real_values,predictions)

# a function that runs a random forest of classification and returns accuracy
def run_class_forest(forest, test_data, start, end, pre):
    predict = []
    while start < end:
        row = test_data.loc[start, :]
        ones = 0
        zeroes = 0
        for tree in forest:
                score = classify(tree, row, pre)
                if score == 1:
                    ones += 1
                elif score == 0:
                    zeroes += 1
        if ones >= zeroes:
            predict.append(1)
        else:
            predict.append(0)
        start += 1
    real_values = test_data[pre].tolist()
    accuracy = met.accuracy_score(real_values, predict)
    return accuracy

# a function that runs a random forest of regression and returns mse
def run_reg_forest(forest, test_data, start, end, pred):
        predict = []
        while start < end:
            row = test_data.loc[start, :]
            sum = 0
            for tree in forest:
                sum += regression(tree, row, pred)
            predict.append(sum / len(forest))
            start += 1
        real_values = test_data[pred].tolist()
        mse = met.mean_squared_error(real_values, predict)
        return mse

# a function that runs a random forest of multyclassification and returns accuracy
def run_multy_class_forest(forest, test_data, start, end, pre):
    predict = []
    curr_vals = sorted(list(set(data[pre])))
    while start < end:
        count = [0] * len(curr_vals)
        row = test_data.loc[start, :]
        for tree in forest:
                score = classify(tree, row, pre)
                count[curr_vals.index(score)] += 1
        max = np.max(count)
        index = count.index(max)
        predict.append(curr_vals[index])
        start += 1
    real_values = test_data[pre].tolist()
    accuracy = met.accuracy_score(real_values, predict)
    return accuracy
# a function that can save a tree or a forest
def save_data(name, object):
    with open(name, 'wb') as output:
        p.dump(object, output, p.HIGHEST_PROTOCOL)

# a function that can read a tree or a forest
def read_data(name):
    with open(name, 'rb') as input:
        object = p.load(input)
    return object
# a function to optimize classification tree with list of max depths and a list of minimum data in a node
def optimize_class_tree(max_depths, min_factors,predict):
    a.featVal(True)
    isFirst = True
    for depth in max_depths:
        for factor in min_factors:
            root = a.create_classification_tree(train_data, depth, False, a.feature_vals, a.feature_numeric_vals, len(a.feature_names),predict)
            tree = a.expand_classification_tree(root, factor, False, a.feature_vals, a.feature_numeric_vals, len(a.feature_names),predict)
            acc = calc_accuracy(validation_data, tree, split1, split2,predict)
            print("finished tree")
            print("accuracy of forest is: " + str(acc))
            print("factor is: " + str(factor))
            print("depth num is: " + str(depth))
            if isFirst:
                best_accuracy = acc
                best_depth = depth
                best_factor = factor
                best_tree = tree
                isFirst = False
            else:
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_depth = depth
                    best_factor = factor
                    best_tree = tree
    return best_tree, best_accuracy, best_depth, best_factor

# a function to optimize multyclassification tree with list of max depths and a list of minimum data in a node
def optimize_multy_class_tree(max_depths, min_factors,predict):
    a.featVal(False)
    isFirst = True
    for depth in max_depths:
        for factor in min_factors:
            root = a.create_multy_classification_tree(train_data, depth, False, a.feature_vals, a.feature_numeric_vals,
                                                      len(a.feature_names), predict)
            tree = a.expand_multy_classification_tree(root, factor, False, a.feature_vals, a.feature_numeric_vals,
                                                      len(a.feature_names), predict)
            acc = calc_multy_accuracy(validation_data, tree, split1, split2,predict)
            print("finished tree")
            print("accuracy of tree is: " + str(acc))
            print("factor is: " + str(factor))
            print("depth num is: " + str(depth))
            if isFirst:
                best_accuracy = acc
                best_depth = depth
                best_factor = factor
                best_tree = tree
                isFirst = False
            else:
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_depth = depth
                    best_factor = factor
                    best_tree = tree
    return best_tree, best_accuracy, best_depth, best_factor

# a function to optimize regression tree with list of max depths and a list of minimum data in a node
def optimize_reg_tree(max_depths, min_factors,predict):
    a.featVal(False)
    isFirst = True
    for depth in max_depths:
        for factor in min_factors:
            root = a.create_regression_tree(train_data, depth, False, a.feature_vals, a.feature_numeric_vals, len(a.feature_names))
            tree = a.expand_regression_tree(root, factor, False, a.feature_vals, a.feature_numeric_vals, len(a.feature_names))
            mse = calc_pred_mse(validation_data, tree, split1, split2,predict)
            print("finished tree")
            print("mse of tree is: " + str(mse))
            print("factor is: " + str(factor))
            print("depth is: " + str(depth))
            if isFirst:
                best_mse = mse
                best_depth = depth
                best_factor = factor
                best_tree = tree
                isFirst = False
            else:
                if mse < best_mse:
                    best_mse = mse
                    best_depth = depth
                    best_factor = factor
                    best_tree = tree
    return best_tree, best_mse, best_depth, best_factor
# a function to optimize classification forest with lists of number of trees, depths, min factors in a node
def optimize_class_forest(num_of_trees, depths, min_factors, predict):
    a.featVal(True)
    isFirst = True
    for tree_number in num_of_trees:
        for depth in depths:
            for factor in min_factors:
                forest = a.build_class_forest(tree_number, 4, factor, depth, a.feature_vals, a.feature_numeric_vals, np.ceil(np.sqrt(len(a.feature_names))),predict)
                accuracy = run_class_forest(forest, validation_data, split1, split2,predict)
                print("finished forest")
                print("acc of forest is: " + str(accuracy))
                print("tree num is: " + str(tree_number))
                print("factor is: " + str(factor))
                print("depth = is: " + str(depth))
                if isFirst:
                    best_acc = accuracy
                    best_forest = forest
                    best_num_of_trees = tree_number
                    best_depth = depth
                    best_min = factor
                else:
                    if best_acc < accuracy:
                        best_acc = accuracy
                        best_forest = forest
                        best_num_of_trees = tree_number
                        best_depth = depth
                        best_min = factor
    return best_forest, best_acc, best_num_of_trees, best_depth, best_min
# a function to optimize multyclassification forest with lists of number of trees, depths, min factors in a node
def optimize_multy_class_forest(num_of_trees, depths, min_factors,predict):
    a.featVal(False)
    isFirst = True
    for tree_number in num_of_trees:
        for depth in depths:
            for factor in min_factors:
                forest = a.build_multyclass_forest(tree_number, 4, factor, depth, a.feature_vals, a.feature_numeric_vals, np.ceil(np.sqrt(len(a.feature_names))),predict)
                accuracy = run_multy_class_forest(forest, validation_data, split1, split2,predict)
                print("finished forest")
                print("acc of forest is: " + str(accuracy))
                print("tree num is: " + str(tree_number))
                print("factor is: " + str(factor))
                print("depth = is: " + str(depth))
                if isFirst:
                    best_acc = accuracy
                    best_forest = forest
                    best_num_of_trees = tree_number
                    best_depth = depth
                    best_min = factor
                else:
                    if best_acc < accuracy:
                        best_acc = accuracy
                        best_forest = forest
                        best_num_of_trees = tree_number
                        best_depth = depth
                        best_min = factor
    return best_forest, best_acc, best_num_of_trees, best_depth, best_min

# a function to optimize regression forest with lists of number of trees, depths, min factors in a node
def optimize_reg_forest(num_of_trees, depths, min_factors,predict):
    a.featVal(False)
    isFirst = True
    for tree_number in num_of_trees:
        for depth in depths:
            for factor in min_factors:
                forest = a.build_reg_forest(tree_number, 4, factor, depth, a.feature_vals, a.feature_numeric_vals, np.ceil(np.sqrt(len(a.feature_names))))
                mse = run_reg_forest(forest, validation_data, split1, split2,predict)
                print("finished forest")
                print("mse of forest is: " + str(mse))
                print("tree num is: " + str(tree_number))
                print("factor is: " + str(factor))
                print("depth = is: " + str(depth))
                if isFirst:
                    best_mse = mse
                    best_forest = forest
                    best_num_of_trees = tree_number
                    best_depth = depth
                    best_min = factor
                else:
                    if best_mse > mse:
                        best_mse = mse
                        best_forest = forest
                        best_num_of_trees = tree_number
                        best_depth = depth
                        best_min = factor
    return best_forest, best_mse, best_num_of_trees, best_depth, best_min

# a method that calculate accuracy of a tree
def calc_multy_accuracy(test_data, tree, start, end, predict):
    count_success = 0
    while start < end:
        row = test_data.loc[start,:]
        true_val = row[predict]
        score = multy_classify(tree, row, predict)
        if true_val == score:
            count_success += 1
        start += 1
    return count_success/len(test_data)

# a method that run multyclassification tree
def multy_classify(tree,row, predict):
    if tree is None:
        return
    if tree.left is None and tree.right is None:
        return tree.multyClassify(predict)
    #categorial
    if tree.is_numerical(a.feature_numeric_vals,a.feature_vals) is False:
         if row[tree.feature]==tree.instance:
            value = multy_classify(tree.left,row, predict)
         else:
            value = multy_classify(tree.right, row, predict)
    #numerical
    else:
        if row[tree.feature] >= tree.instance:
            value = multy_classify(tree.left,row, predict)
        else:
            value = multy_classify(tree.right, row, predict)
    return value




# run a classification tree with optimization parameters: [depths], [min factors]
tree, accuracy, depth, factor = optimize_class_tree([6,7,8],[2,3,4],'>50K')
print(calc_accuracy(test_data,tree,split2,len(data),'>50K'))
print("depth:")
print(str(depth))
print("factor:")
print(str(factor))

# run a classification forest with optimization parameters: [number of trees], [depths], [min factors]
forest, accuracy, tree_num, depth, min_factor = optimize_class_forest([50,100],[6,7],[2,3],'>50K')
print("optimal values:")
print("tree num:")
print(str(tree_num))
print("depth: ")
print(str(depth))
print("min factor: ")
print(str(min_factor))
print(run_class_forest(forest, test_data, split2, len(data),'>50K'))

# run a multyclassification tree with optimization parameters: [depths], [min factors]
tree, accuracy, depth, factor = optimize_multy_class_tree([6,7,8],[2,3,4],'education')
print("optimal values:")
print(accuracy)
print(depth)
print(factor)
print(calc_multy_accuracy(test_data,tree,split2,len(data),'education'))

# run a multyclassification forest with optimization parameters: [number of trees], [depths], [min factors]
forest, accuracy, tree_num, depth, factor = optimize_multy_class_forest([50,100],[6,7],[2,3], "education")
print("optimal values:")
print("tree num:")
print(str(tree_num))
print("depth: ")
print(str(depth))
print("min factor: ")
print(str(factor))
print(run_multy_class_forest(forest, test_data, split2, len(data),"education"))

# run a regression tree with optimization parameters: [depths], [min factors]
tree, mse, depth, factor = optimize_reg_tree([6,7,8],[2,3,4],'education-num')
save_data("msetree",tree)
print("depth:")
print(str(depth))
print("factor:")
print(str(factor))
print(calc_pred_mse(test_data,tree, split2,len(data),'education-num'))



# run a regression forest with optimization parameters: [number of trees], [depths], [min factors]
forest, mse, tree_num, depth, factor = optimize_reg_forest([50,100], [6,7], [2,3] ,'education-num')
save_data("mseForest",forest)
print("optimal values:")
print("tree num:")
print(str(tree_num))
print("depth: ")
print(str(depth))
print("min factor: ")
print(str(factor))
print(run_reg_forest(forest, test_data, split2,len(data),'education-num'))
