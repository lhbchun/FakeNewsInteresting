import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

global covidRumor, constraintAAAI, offComm, coaid, truthseeker

from verboseLogger import Logger

# sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    TfidfTransformer,
    CountVectorizer,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression


def getParameters(df):
    gispyParam = [
        "DESPC",
        "DESSC",
        "CoREF",
        "PCREF_1",
        "PCREF_a",
        "PCREF_1p",
        "PCREF_ap",
        "PCDC",
        "SMCAUSe_1",
        "SMCAUSe_a",
        "SMCAUSe_1p",
        "SMCAUSe_ap",
        "SMCAUSwn_1p_path",
        "SMCAUSwn_1p_lch",
        "SMCAUSwn_1p_wup",
        "SMCAUSwn_ap_path",
        "SMCAUSwn_ap_lch",
        "SMCAUSwn_ap_wup",
        "SMCAUSwn_1_path",
        "SMCAUSwn_1_lch",
        "SMCAUSwn_1_wup",
        "SMCAUSwn_a_path",
        "SMCAUSwn_a_lch",
        "SMCAUSwn_a_wup",
        "SMCAUSwn_1p_binary",
        "SMCAUSwn_ap_binary",
        "SMCAUSwn_1_binary",
        "SMCAUSwn_a_binary",
        "PCCNC_megahr",
        "WRDIMGc_megahr",
        "PCCNC_mrc",
        "WRDIMGc_mrc",
        "WRDHYPnv",
        "gis",
    ]
    otherParam = [
        "sentiment_1",
        "sentiment_2",
        "sentiment_3",
        "sentiment_4",
        "sentiment_5",
        "valence",
        "arousal",
        "dominance",
        "MaxRank",
        "MedianRank",
        "MedianContentRank",
        "MinContentRank",
        "roberta_arousal",
        "roberta_dominance",
        "roberta_concreteness",
    ]
    otherParam.extend(gispyParam)

    return df[otherParam]


def getParametersAndText(df, textCol):
    gispyParam = [
        "DESPC",
        "DESSC",
        "CoREF",
        "PCREF_1",
        "PCREF_a",
        "PCREF_1p",
        "PCREF_ap",
        "PCDC",
        "SMCAUSe_1",
        "SMCAUSe_a",
        "SMCAUSe_1p",
        "SMCAUSe_ap",
        "SMCAUSwn_1p_path",
        "SMCAUSwn_1p_lch",
        "SMCAUSwn_1p_wup",
        "SMCAUSwn_ap_path",
        "SMCAUSwn_ap_lch",
        "SMCAUSwn_ap_wup",
        "SMCAUSwn_1_path",
        "SMCAUSwn_1_lch",
        "SMCAUSwn_1_wup",
        "SMCAUSwn_a_path",
        "SMCAUSwn_a_lch",
        "SMCAUSwn_a_wup",
        "SMCAUSwn_1p_binary",
        "SMCAUSwn_ap_binary",
        "SMCAUSwn_1_binary",
        "SMCAUSwn_a_binary",
        "PCCNC_megahr",
        "WRDIMGc_megahr",
        "PCCNC_mrc",
        "WRDIMGc_mrc",
        "WRDHYPnv",
        "gis",
    ]
    otherParam = [
        "sentiment_1",
        "sentiment_2",
        "sentiment_3",
        "sentiment_4",
        "sentiment_5",
        "valence",
        "arousal",
        "dominance",
        "MaxRank",
        "MedianRank",
        "MedianContentRank",
        "MinContentRank",
        "roberta_arousal",
        "roberta_dominance",
        "roberta_concreteness",
    ]
    otherParam.extend(gispyParam)
    otherParam.append("content")
    df.rename(columns={textCol: "content"}, inplace=True)

    return df[otherParam]


def labelThisAs(df, label):
    return pd.concat([df, pd.DataFrame([label] * len(df), columns=["label"])], axis=1)


def initDataset(includeText=False, includeLabel=False, returnList=False):

    initRaw()
    global a, b, c, d, e, f, g, h, listData

    textColName = [
        "content",
        "content",
        "tweet",
        "tweet",
        "rawContent",
        "rawContent",
        "tweet",
        "tweet",
    ]
    listLabel = [
        "fake",
        "non-fake",
        "fake",
        "non-fake",
        "offComm",
        "fake",
        "fake",
        "non-fake",
    ]

    if includeText:
        for i in range(len(listData)):
            listData[i] = getParametersAndText(listData[i], textColName[i])
    else:
        for i in range(len(listData)):
            listData[i] = getParameters(listData[i])

    if includeLabel:
        for i in range(len(listData)):
            listData[i] = labelThisAs(listData[i], listLabel[i])

    a = listData[0]
    b = listData[1]
    c = listData[2]
    d = listData[3]
    e = listData[4]
    f = listData[5]
    g = listData[6]
    h = listData[7]

    if returnList:
        return listData

    a = covidRumor[covidRumor["label"] == "F"].reset_index(drop=True)
    b = covidRumor[covidRumor["label"] == "T"].reset_index(drop=True)
    c = constraintAAAI[constraintAAAI["label"] == "fake"].reset_index(drop=True)
    d = constraintAAAI[constraintAAAI["label"] == "real"].reset_index(drop=True)
    e = offComm
    f = coaid
    g = truthseeker[~truthseeker["target"]].reset_index(drop=True)
    h = truthseeker[truthseeker["target"]].reset_index(drop=True)

    listData = [a, b, c, d, e, f, g, h]


def preprocess(df, dropna=True):

    df.loc[
        :, ["MaxRank", "MedianRank", "MedianContentRank", "MinContentRank"]
    ] = np.log(
        df.loc[:, ["MaxRank", "MedianRank", "MedianContentRank", "MinContentRank"]]
    )
    df["Sentiment"] = (
        0.25 * df["sentiment_2"]
        + 0.5 * df["sentiment_3"]
        + 0.75 * df["sentiment_4"]
        + 1 * df["sentiment_5"]
    )
    df = df.drop(columns=["sentiment_" + str(i + 1) for i in range(5)])
    if dropna:
        df = df.dropna()
    return df


def sentiHist(df):

    fig, ax = plt.subplots(nrows=2, ncols=3)
    ax[0][0].hist(df["sentiment_1"], bins=10, range=[0, 1])
    ax[0][0].set_title("Sentiment = 1")

    ax[0][1].hist(df["sentiment_2"], bins=10, range=[0, 1])
    ax[0][1].set_title("Sentiment = 2")

    ax[0][2].hist(df["sentiment_3"], bins=10, range=[0, 1])
    ax[0][2].set_title("Sentiment = 3")

    ax[1][0].hist(df["sentiment_4"], bins=10, range=[0, 1])
    ax[1][0].set_title("Sentiment = 4")

    ax[1][1].hist(df["sentiment_5"], bins=10, range=[0, 1])
    ax[1][1].set_title("Sentiment = 5")

    ax[1][2].hist(
        0.25 * df["sentiment_2"]
        + 0.5 * df["sentiment_3"]
        + 0.75 * df["sentiment_4"]
        + df["sentiment_5"],
        bins=10,
        range=[0, 1],
    )
    ax[1][2].set_title("Expected")

    for i in range(2):
        for j in range(3):
            ax[i][j].tick_params("both", labelsize=6, width=0.2)
            ax[i][j].xaxis.set_ticks([0.2 * i for i in range(6)])

    plt.show()


def WordRankHist(df, name):

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0][0].hist(np.log(df["MaxRank"]), bins=[i for i in range(15)], density=True)
    ax[0][0].set_title("Max")

    ax[0][1].hist(np.log(df["MedianRank"]), bins=[i for i in range(15)], density=True)
    ax[0][1].set_title("Median")

    ax[1][0].hist(
        np.log(df["MedianContentRank"]), bins=[i for i in range(15)], density=True
    )
    ax[1][0].set_title("Median Content")

    ax[1][1].hist(
        np.log(df["MinContentRank"]), bins=[i for i in range(15)], density=True
    )
    ax[1][1].set_title("Min Content")

    for i in range(2):
        for j in range(2):
            ax[i][j].tick_params("both", labelsize=6, width=0.2)
            ax[i][j].xaxis.set_ticks([i for i in range(2, 15, 1)])
            ax[i][j].yaxis.set_ticks([i * 0.1 for i in range(0, 8, 1)])
            ax[i][j].set_xlim([2, 15])
            ax[i][j].set_ylim([0, 0.7])

    fig.suptitle(name)

    plt.show()


def ConcretenessHist(df, name):

    plt.hist(
        df["roberta_concreteness"],
        bins=[i / 10.0 for i in range(-20, 20)],
        density=True,
    )
    plt.title("Max")
    plt.xlim([-2, 2])
    plt.ylim([0, 1.2])
    plt.title(name)

    plt.show()


def getFeatureMetrics(X, y):

    binary = True
    df = []

    for colName in X.columns:
        dtc = DecisionTreeClassifier(criterion="gini", max_depth=1, random_state=928)
        dtc.fit([[i] for i in X[colName].to_list()], y)
        n = dtc.tree_.n_node_samples
        impurity = dtc.tree_.impurity
        if len(n) == 1:
            df.append(
                [
                    colName,
                    0,
                    None,
                    None,
                    accuracy_score(dtc.predict([[i] for i in X[colName].to_list()]), y),
                ]
            )
        else:
            result = []
            result.extend(
                [
                    colName,
                    impurity[0]
                    - sum([n[i] / n[0] * impurity[i] for i in range(1, len(n))]),
                ]
            )

            # append left majority
            result.append(dtc.classes_[np.argmax(dtc.tree_.value[1][0])])
            # append right majority
            result.append(dtc.classes_[np.argmax(dtc.tree_.value[2][0])])

            yHat = dtc.predict([[i] for i in X[colName].to_list()])
            result.append(accuracy_score(y, yHat))
            result.append(balanced_accuracy_score(y, yHat))

            yHat = dtc.predict_proba([[i] for i in X[colName].to_list()])[:, 1]
            result.append(roc_auc_score(y, yHat))

            df.append(result)

    return pd.DataFrame(
        df,
        columns=[
            "feature",
            "gain",
            "left",
            "right",
            "accuracy",
            "accuracy (balanced)",
            "auc",
        ],
    ).sort_values("gain", ascending=False)


def isVacRelated(text):
    return bool(
        re.search(
            pattern=".*(vaccine|vaccination|vaccinated|booster|dose).*", string=text
        )
    )


def isInfecRelated(text):
    return bool(
        re.search(
            pattern=".*(case|spread|death|total number|positive|infect).*", string=text
        )
    )


def isPrevRelated(text):
    return bool(
        re.search(pattern=".*(protect|prevent|wash|face covering).*", string=text)
    )


def isTestRelated(text):
    return bool(re.search(pattern=".*(test).*", string=text))


def isSocEventRelated(text):
    return bool(re.search(pattern=".*(work|school|lockdown).*", string=text))


def isSymptomRelated(text):
    return bool(re.search(pattern=".*(symptom|cough).*", string=text))


def KLTransform(x, nComponents=None):

    X = x.copy()
    procX = X - np.mean(X, axis=0)
    procX = np.dot(procX.T, procX) / X.shape[0]
    pca = PCA()
    pca.fit(procX)

    lambdas = []
    # eigenvector and eigenvalues
    for v in pca.components_:
        lambdas.append(np.dot(v.T, np.dot(procX, v)))
    _, sortedPhi = zip(
        *sorted(zip(lambdas, pca.components_), key=lambda x: x[0], reverse=True)
    )
    sortedPhi = np.array(sortedPhi)
    result = np.dot(sortedPhi, X.T).T

    if nComponents != None:
        return np.array([i[len(i) - nComponents :] for i in result])

    return result


def KLT(x, nComponents=None):

    x.loc[:, ["MaxRank", "MedianRank", "MedianContentRank", "MinContentRank"]] = np.log(
        x.loc[:, ["MaxRank", "MedianRank", "MedianContentRank", "MinContentRank"]]
    )
    x["Sentiment"] = (
        0.25 * x["sentiment_2"]
        + 0.5 * x["sentiment_3"]
        + 0.75 * x["sentiment_4"]
        + 1 * x["sentiment_5"]
    )

    x = (
        x.drop(columns=["sentiment_" + str(i + 1) for i in range(5)])
        .dropna()
        .reset_index()
    )
    y = None
    z = None

    try:
        y = x["label"]
        x = x.drop(columns=["label"])
    except KeyError:
        pass

    try:
        z = x["content"]
        x = x.drop(columns=["content"])
    except KeyError:
        pass

    return KLTransform(x, nComponents), y, z


def express(logger: Logger):
    covidRumor = pd.read_csv("data/preprocessed/covidRumor.csv", index_col=0)
    constraintAAAI = pd.read_csv("data/preprocessed/constraintAAAI.csv", index_col=0)
    offComm = pd.read_csv("data/preprocessed/officalComm.csv", index_col=0)
    coaid = pd.read_csv("data/preprocessed/coaid.csv", index_col=0)
    truthseeker = pd.read_csv("data/preprocessed/truthseeker.csv", index_col=0)
    initDataset()

    sentiHist(covidRumor)
    sentiHist(covidRumor[covidRumor["label"] == "F"])
    sentiHist(covidRumor[covidRumor["label"] == "T"])
    sentiHist(covidRumor[covidRumor["label"] == "U"])
    sentiHist(constraintAAAI[constraintAAAI["label"] == "fake"])
    sentiHist(constraintAAAI[constraintAAAI["label"] == "real"])

    WordRankHist(offComm, "Word Rank for Official Communication")
    WordRankHist(covidRumor, "Word Rank for Rumors")
    WordRankHist(covidRumor[covidRumor["label"] == "F"], "Word Rank for Fake Rumors")
    WordRankHist(covidRumor[covidRumor["label"] == "T"], "Word Rank for True Rumors")
    WordRankHist(covidRumor[covidRumor["label"] == "U"], "Word Rank for Unknown Rumors")
    WordRankHist(
        constraintAAAI[constraintAAAI["label"] == "fake"],
        "Word Rank for Fake News AAAI dataset",
    )
    WordRankHist(
        constraintAAAI[constraintAAAI["label"] == "real"],
        "Word Rank for True News AAAI dataset",
    )
    WordRankHist(constraintAAAI, "Word Rank for AAAI dataset")

    ConcretenessHist(covidRumor[covidRumor["label"] == "F"], "covid rumor conreteness")
    ConcretenessHist(
        covidRumor[covidRumor["label"] == "T"], "covid rumor (true) conreteness"
    )
    ConcretenessHist(
        constraintAAAI[constraintAAAI["label"] == "real"], "real news conreteness"
    )
    ConcretenessHist(
        constraintAAAI[constraintAAAI["label"] == "fake"], "fake news conreteness"
    )
    ConcretenessHist(offComm, "official communication conreteness")

    dtc = DecisionTreeClassifier(criterion="gini", max_depth=1, random_state=928)
    dtc.fit([[i] for i in xTrain["Sentiment"].to_list()], yTrain)
    dtc.tree_.n_node_samples

    getFeatureMetrics(xTrain, yTrain)

    initDataset(includeText=False, includeLabel=True)

    xy2 = pd.concat([e])
    xy1 = resample(
        pd.concat([a, c, f, g]), replace=True, n_samples=len(xy2), random_state=928
    )

    xy = pd.concat([xy1, xy2])

    xy.loc[
        :, ["MaxRank", "MedianRank", "MedianContentRank", "MinContentRank"]
    ] = np.log(
        xy.loc[:, ["MaxRank", "MedianRank", "MedianContentRank", "MinContentRank"]]
    )
    xy["Sentiment"] = (
        0.25 * xy["sentiment_2"]
        + 0.5 * xy["sentiment_3"]
        + 0.75 * xy["sentiment_4"]
        + 1 * xy["sentiment_5"]
    )
    xy = xy.drop(columns=["sentiment_" + str(i + 1) for i in range(5)]).dropna()

    Y = xy["label"]
    X = xy.drop(columns=["label"])
    train_test_split

    xTrain, xTest, yTrain, yTest = train_test_split(
        X, Y, test_size=0.2, random_state=928
    )

    dtc1 = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=4,
        random_state=928,
    )
    dtc1.fit(xTrain, yTrain)
    logger.log("Training accuracy: " + str(accuracy_score(dtc1.predict(xTrain), yTrain)))
    logger.log("Testing accuracy: " + str(accuracy_score(dtc1.predict(xTest), yTest)))

    # pd.concat([
    #     pd.DataFrame(xTrain.columns.tolist(), columns = ["Name"]),
    #     pd.DataFrame(dtc1.feature_importances_, columns = ["Importance"])
    # ], axis = 1).sort_values("Importance", ascending=False)

    getFeatureMetrics(xTrain, yTrain)

    ## LGBM
    lgb1 = LGBMClassifier(n_estimators=1000, importance_type="gain", random_state=928)
    lgb1.fit(xTrain, yTrain)
    logger.log("Training accuracy: " + str(accuracy_score(lgb1.predict(xTrain), yTrain)))
    logger.log("Testing accuracy: " + str(accuracy_score(lgb1.predict(xTest), yTest)))

    initDataset(includeText=False, includeLabel=True)

    xy1 = pd.concat([a, c, f, g])
    xy2 = resample(
        pd.concat([b, d, h]), replace=True, n_samples=len(xy1), random_state=928
    )
    xy = pd.concat([xy1, xy2])

    xy.loc[
        :, ["MaxRank", "MedianRank", "MedianContentRank", "MinContentRank"]
    ] = np.log(
        xy.loc[:, ["MaxRank", "MedianRank", "MedianContentRank", "MinContentRank"]]
    )
    xy["Sentiment"] = (
        0.25 * xy["sentiment_2"]
        + 0.5 * xy["sentiment_3"]
        + 0.75 * xy["sentiment_4"]
        + 1 * xy["sentiment_5"]
    )
    xy = xy.drop(columns=["sentiment_" + str(i + 1) for i in range(5)]).dropna()

    Y = xy["label"]
    X = xy.drop(columns=["label"])
    train_test_split

    xTrain, xTest, yTrain, yTest = train_test_split(
        X, Y, test_size=0.2, random_state=928
    )

    dtc2 = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=4,
        random_state=928,
    )
    dtc2.fit(xTrain, yTrain)

    logger.log("Training accuracy: " + str(accuracy_score(dtc2.predict(xTrain), yTrain)))
    logger.log("Testing accuracy: " + str(accuracy_score(dtc2.predict(xTest), yTest)))
    # pd.concat([
    #     pd.DataFrame(xTrain.columns.tolist(), columns = ["Name"]),
    #     pd.DataFrame(dtc2.feature_importances_, columns = ["Importance"])
    # ], axis = 1).sort_values("Importance", ascending=False)

    getFeatureMetrics(xTrain, yTrain)

    ada2 = AdaBoostClassifier(n_estimators=1000, random_state=928)
    ada2.fit(xTrain, yTrain)

    logger.log("Training accuracy: " + str(accuracy_score(ada2.predict(xTrain), yTrain)))
    logger.log("Testing accuracy: " + str(accuracy_score(ada2.predict(xTest), yTest)))

    ## LGBM
    lgb2 = LGBMClassifier(n_estimators=1000, importance_type="gain", random_state=928)
    lgb2.fit(xTrain, yTrain)

    logger.log("Training accuracy: " + str(accuracy_score(lgb2.predict(xTrain), yTrain)))
    logger.log("Testing accuracy: " + str(accuracy_score(lgb2.predict(xTest), yTest)))

    initDataset(includeText=False, includeLabel=True)
    xy = pd.concat([a, b, c, d, e, f, g, h])

    xy.loc[
        :, ["MaxRank", "MedianRank", "MedianContentRank", "MinContentRank"]
    ] = np.log(
        xy.loc[:, ["MaxRank", "MedianRank", "MedianContentRank", "MinContentRank"]]
    )
    xy["Sentiment"] = (
        0.25 * xy["sentiment_2"]
        + 0.5 * xy["sentiment_3"]
        + 0.75 * xy["sentiment_4"]
        + 1 * xy["sentiment_5"]
    )
    xy = xy.drop(columns=["sentiment_" + str(i + 1) for i in range(5)]).dropna()

    Y = xy["label"]
    X = xy.drop(columns=["label"])
    train_test_split

    xTrain, xTest, yTrain, yTest = train_test_split(
        X, Y, test_size=0.001, random_state=928
    )

    dtc3 = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=4,
        random_state=928,
    )
    dtc3.fit(xTrain, yTrain)

    logger.log("Training accuracy: " + str(accuracy_score(dtc3.predict(xTrain), yTrain)))
    logger.log("Testing accuracy: " + str(accuracy_score(dtc3.predict(xTest), yTest)))
    pd.concat(
        [
            pd.DataFrame(xTrain.columns.tolist(), columns=["Name"]),
            pd.DataFrame(dtc3.feature_importances_, columns=["Importance"]),
        ],
        axis=1,
    ).sort_values("Importance", ascending=False)

    ada3 = AdaBoostClassifier(n_estimators=1000, random_state=928)
    ada3.fit(xTrain, yTrain)

    logger.log("Training accuracy: " + str(accuracy_score(ada3.predict(xTrain), yTrain)))
    logger.log("Testing accuracy: " + str(accuracy_score(ada3.predict(xTest), yTest)))
    pd.concat(
        [
            pd.DataFrame(xTrain.columns.tolist(), columns=["Name"]),
            pd.DataFrame(ada3.feature_importances_, columns=["Importance"]),
        ],
        axis=1,
    ).sort_values("Importance", ascending=False)

    ## LGBM
    lgb3 = LGBMClassifier(n_estimators=1000, importance_type="gain", random_state=928)
    lgb3.fit(xTrain, yTrain)

    logger.log("Training accuracy: " + str(accuracy_score(lgb3.predict(xTrain), yTrain)))
    logger.log("Testing accuracy: " + str(accuracy_score(lgb3.predict(xTest), yTest)))
    pd.concat(
        [
            pd.DataFrame(xTrain.columns.tolist(), columns=["Name"]),
            pd.DataFrame(lgb3.feature_importances_, columns=["Importance"]),
        ],
        axis=1,
    ).sort_values("Importance", ascending=False)

    xy = pd.concat([a, b, c, d, e, f, g, h])
    xy.dropna()
    tr, Y0, Z0 = KLT(xy, 10)

    relationFunction = [
        isVacRelated,
        isInfecRelated,
        isPrevRelated,
        isTestRelated,
        isSocEventRelated,
        isSymptomRelated,
    ]

    vacRelated = np.array([isVacRelated(i) for i in Z0])
    infecRelated = np.array([isInfecRelated(i) for i in Z0])
    prevRelated = np.array([isPrevRelated(i) for i in Z0])
    testRelated = np.array([isTestRelated(i) for i in Z0])
    socEventRelated = np.array([isSocEventRelated(i) for i in Z0])
    symptomRelated = np.array([isSymptomRelated(i) for i in Z0])

    listData = initDataset(includeText=True, includeLabel=True, returnList=True)

    contentTopics = []

    for i in range(8):
        temp = []
        for j in range(6):
            temp.append([relationFunction[j](k) for k in listTopics[i]])
        contentTopics.append(temp)

    logger.log("vaccine related percentage: " + str(round(vacRelated.sum() / len(Z0), 4)))
    logger.log(
        "infection related percentage: " + str(round(infecRelated.sum() / len(Z0), 4))
    )
    logger.log(
        "prevention related percentage: " + str(round(prevRelated.sum() / len(Z0), 4))
    )
    logger.log("test related percentage: " + str(round(testRelated.sum() / len(Z0), 4)))
    logger.log(
        "social event related percentage: "
        + str(round(socEventRelated.sum() / len(Z0), 4))
    )
    logger.log(
        "sympton related percentage: " + str(round(symptomRelated.sum() / len(Z0), 4))
    )

    covered = np.logical_or(
        np.logical_or(
            np.logical_or(
                np.logical_or(np.logical_or(vacRelated, infecRelated), prevRelated),
                testRelated,
            ),
            socEventRelated,
        ),
        symptomRelated,
    )
    logger.log("covered percentage: " + str(round(covered.sum() / len(Z0), 4)))

    topics = [
        vacRelated,
        infecRelated,
        prevRelated,
        testRelated,
        socEventRelated,
        symptomRelated,
    ]
    topicName = [
        "vaccine",
        "infection",
        "prevention",
        "testing",
        "social events",
        "symptoms",
    ]
    matrix = [[0 for i in range(6)] for i in range(6)]

    for i in range(len(topics)):
        for j in range(len(topics)):
            matrix[i][j] = round(
                np.logical_and(topics[i], topics[j]).sum() / len(Z0), 4
            )

    pd.DataFrame(matrix, columns=topicName, index=topicName)

    listData = initDataset(includeText=True, includeLabel=True, returnList=True)
    result = []
    topics = [
        isVacRelated,
        isInfecRelated,
        isPrevRelated,
        isTestRelated,
        isSocEventRelated,
        isSymptomRelated,
    ]
    for dat in listData:
        _, _, Z0 = KLT(dat, 10)
        # Z0 = Z0.dropna().reset_index(drop = True)
        related = []
        for j in topics:
            related.append(np.array([j(k) for k in Z0]))
        result.append(related)
    logger.log(result)

    pd.DataFrame(
        np.array([[round(i.sum() / len(i), 4) for i in j] for j in result]).T,
        columns=[
            "covidRumor - Fake",
            "covidRumor - NonFake",
            "AAAI - Fake",
            "AAAI - NonFake",
            "OffComm",
            "COAID",
            "TruthSeeker - Fake",
            "TruthSeeker - NonFake",
        ],
        index=topicName,
    )
