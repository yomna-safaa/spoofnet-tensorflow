
import numpy as np

###################################################################
### accracy given scores + threshold:
def Accuracy(prediction_scores, true_labels, threshold):
    true_labels = np.asarray(true_labels, dtype='int64')
    # pos = real = 1
    # neg = attack = fake = 0
    scores_results = np.empty(len(prediction_scores),dtype='int64')
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, s in enumerate(prediction_scores):
        if s > threshold:
            scores_results[i] = 1  # predicted as positive: tp or fp?
            if true_labels[i] == 1:  # originally pos #pos = real = 1
                tp += 1
            else:  # originally neg
                fp += 1
        else:  # predicted as negative: tn or fn? # neg = attack = fake = 0
            scores_results[i] = 0
            if true_labels[i] == 1:  # originally pos
                fn += 1
            else:  # originally neg
                tn += 1

    print ("Accuracy calculated at threshold: ", threshold)
    correctResults = (scores_results == true_labels).sum()
    print ("Correctly Classified: ", correctResults, ", out of: ", len(true_labels), " images.")
    acc = correctResults * 1.0 / len(true_labels)
    print ("Model accuracy (%): ", acc * 100, "%")

    nPos = tp + fn
    nNeg = fp + tn

    far = fp * 1.0 / nNeg
    frr = fn * 1.0 / nPos
    hter = (far + frr) * 100.0 / 2

    acc = (tp + tn) * 1.0 / (nPos + nNeg) # + 1.0e-10)
    print ("Accuracy (%): ", acc * 100, "%\nFAR: ", far, ", FRR: ", frr, ', HTER: (%)', hter, "%")

    return acc, far, frr, hter


import matplotlib.pyplot as plt
def roc_det(valScores_predicted, valLabels_true, display_details=True):

    ### ROC: fpr vs tpr
    # calculate the fpr and tpr for all thresholds of the classification
    method = 2

    if method ==1:
        from sklearn import metrics
        #### ROC method1:
        preds = valScores_predicted
        y = np.asarray(valLabels_true, dtype='int64')
        y[y == 0] = -1

        fpr, tpr, thresholds = metrics.roc_curve(y, preds)
        roc_auc = metrics.auc(fpr, tpr)
        # print y
        # print fpr
        # print tpr
        # print thresholds
        print (roc_auc)
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')

    ##############
    elif method ==2:

        #### ROC method2:
        ##### ##### Another way to plot roc, using scatter:
        roc_x = [] #fpr
        roc_y = [] #tpr
        min_score = min(valScores_predicted)
        max_score = max(valScores_predicted)
        # step = (max_score - min_score) *1.0/100
        # step = int(np.ceil(step))
        # thresholds = np.linspace(min_score, max_score, step)
        thresholds = np.linspace(min_score, max_score, num=100)
        FP=0
        TP=0
        y = np.asarray(valLabels_true, dtype='int64')
        P = sum(y)
        N = len(y) - P

        # for (j, T) in enumerate(thresholds):
        for (i, T) in enumerate(thresholds):
            for i in range(0, len(valScores_predicted)):
                if (valScores_predicted[i] > T):
                # if (valScores_predicted[i] >= T): #YY: changed this on 10Jan2017 to (>=), since when it was only (>) when threshold ==1, if predSc=1, so it's not >T, and so TP will not be incremented and will stay 0!!!!
                    if (y[i]==1):
                        TP = TP + 1
                    if (y[i]==0):
                        FP = FP + 1
            roc_x.append(FP/float(N))
            roc_y.append(TP/float(P))
            FP=0
            TP=0
        fpr, tpr = roc_x, roc_y
        # plt.scatter(fpr, tpr) #(roc_x, roc_y)
        plt.plot(fpr, tpr)
        fpr = np.asarray(fpr)
        tpr = np.asarray(tpr)

    ###################
    plt.title('Receiver Operating Characteristic')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show(block=False)

    ###################################################################
    ### DET: fpr vs fnr
    ### ROC: fpr vs tpr

    ######### DET::
    fnr = 1 - tpr

    plt.title('DET curve')
    plt.plot(fpr, fnr, 'b')
    # plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('False Negative Rate')
    plt.xlabel('False Positive Rate')

    ###################################################################
    ### EER:
    d = abs(fpr - fnr)
    if display_details:
        print (d)
    # j = d.argmin()
    j = np.where(d == d.min())
    j = j[0]
    # print j, d[j]
    # print fpr[j], fnr[j]
    if len(j)>1:
        candidate_thresholds = thresholds[j]
        print ("Several EER_Thresholds")
        if display_details:
            print (candidate_thresholds)
        eer_thr_i = candidate_thresholds.argmin()
        eer_thr_j = j[eer_thr_i]
        # eer_thr1 = candidate_thresholds[eer_thr_i]
        # eer_thr2 = thresholds[eer_thr_j]
        j = eer_thr_j
        eer = fpr[j]
        eer_thr = thresholds[j]
        eer_thr2 = max(candidate_thresholds)
        if eer_thr<0 and max(candidate_thresholds)>0:
            print ("Several EER_Thresholds, min = ", eer_thr, ", Will set EER_Threshold = 0")
            eer_thr=0
    else:
        eer = fpr[j]
        eer_thr = thresholds[j]
        eer_thr2=-1
    print ('eer, eer_thr, eer_thr(max): ', eer, eer_thr, eer_thr2)

    plt.scatter(fpr[j], fnr[j])

    ###################################################################
    ## Y: try on Nov10 2016, to get thr which has min (fpr+fnr)
    ## mer (min err rate) instead of eer (equal error rate)
    s = fpr + fnr
    if display_details:
        print (s)
    jj = np.where(s == s.min())
    jj = jj[0]
    if len(jj) > 1:
        candidate_thresholds = thresholds[jj]
        print ("Several MER_Thresholds")
        if display_details:
            print (candidate_thresholds)
        mer_thr_ii = candidate_thresholds.argmin()
        mer_thr_jj = jj[mer_thr_ii]
        # eer_thr1 = candidate_thresholds[eer_thr_i]
        # eer_thr2 = thresholds[eer_thr_j]
        jj = mer_thr_jj
        mer = fpr[jj]
        mer_thr = thresholds[jj]
        if mer_thr < 0 and max(candidate_thresholds) > 0:
            print ("Several MER_Thresholds, min = ", mer_thr, ", Will set MER_Threshold = 0")
            mer_thr = 0
    else:
        mer = fpr[jj]
        mer_thr = thresholds[jj]
    print ('mer, mer_thr: ', mer, mer_thr)

    plt.scatter(fpr[jj], fnr[jj])
    plt.show(block=False)

    ###################################################################
    # Accuracy(valScores_predicted, valLabels_true, 0)
    # Accuracy(valScores_predicted, valLabels_true, 0.5)
    print ("\nEER calculated using this Data = ", eer, ", EER_threshold= ", eer_thr)
    Accuracy(valScores_predicted, valLabels_true, eer_thr)
    if abs(mer_thr - eer_thr) > 0:
        print ("\nMER calculated using this Data = ", mer, ", MER_threshold= ", mer_thr)
        Accuracy(valScores_predicted, valLabels_true, mer_thr)

    return eer_thr, mer_thr, eer_thr2
