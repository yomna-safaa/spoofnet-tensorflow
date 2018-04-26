import csv
import os

import data.datasets_paths as datasets_paths

#############################################################################
DB = 'ATVS'
# DB = 'Warsaw'
# DB = 'MobBioFake'

######################################################
if DB == 'ATVS':
    SET_rootFolder = datasets_paths.IRIS_ROOT_DIR + datasets_paths.ATVS_SUB_DIR
    filePre = "/biosec"
    sourceDir = SET_rootFolder + "/ATVS-FIr_DB"

elif DB == 'Warsaw':
    SET_rootFolder = datasets_paths.IRIS_ROOT_DIR + datasets_paths.WARSAW_SUB_DIR
    filePre = "/warsaw"

elif DB == 'MobBioFake':
    SET_rootFolder = datasets_paths.IRIS_ROOT_DIR + datasets_paths.MOBBIOFAKE_SUB_DIR
    filePre = "/mobbiofake"

######################################################
labelFilesDir = SET_rootFolder + "/labelFiles"
trainFile = labelFilesDir + filePre + "_spoof_train.txt"
testfile = labelFilesDir + filePre + "_spoof_test.txt"
categories_file = labelFilesDir + filePre + "_categories.txt"

if not os.path.exists(labelFilesDir):
    os.makedirs(labelFilesDir)

######################################################
sets = ["FAKE", "REAL"]
caffe_sets = ["train", "test"]


#######################################################
def create_labelsTxtFile_ATVS():
    with open(categories_file, 'w') as writeF:
        for s in sets:  # label = index (0: FAKE, 1: REAL)
            writeF.write("%s\n" % s)

    writeF = labelFilesDir + "/biosec_spoof_all.txt"
    with open(writeF, 'wb') as writeF:
        mywr = csv.writer(writeF, delimiter=' ')
        for label, s in enumerate(sets):  # label = index (0: FAKE, 1: REAL)
            setDir = sourceDir + "/" + s
            usersDir = os.listdir(setDir)  ## 50 users
            usersDir.sort()
            for user in usersDir:
                files = os.listdir(setDir + "/" + user)
                for img in files:
                    mywr.writerow([s + '/' + user + '/' + img] + [label])

                ###################
                ### taking 25% for training, and the remaining 75% for testing ..
                ## so first 12.5 users (real and fake) for training .. that is since each user has 2 eyes and each eye is treated as
                ## a diff user, so take :
                # Train: users 1 to 12 + left eye of user 13
                # Test: users 14 to 50 + right eye of user 13
    trainF = trainFile
    with open(trainF, 'wb') as trainF:
        mywr = csv.writer(trainF, delimiter=' ')
        for label, s in enumerate(sets):  # label = index (0: FAKE, 1: REAL)
            setDir = sourceDir + "/" + s
            usersDir = os.listdir(setDir)  ## 50 users
            usersDir.sort()
            for i, user in enumerate(usersDir):
                # Train: users 1 to 12 + left eye of user 13 (i.e: index 0:11, + left of index12
                if i > 12:
                    break
                files = os.listdir(setDir + "/" + user)
                for img in files:
                    if i == 12:
                        if "_l_" in img:
                            mywr.writerow([s + '/' + user + '/' + img] + [label])
                    else:
                        mywr.writerow([s + '/' + user + '/' + img] + [label])

    ###################
    testF = testfile
    with open(testF, 'wb') as testF:
        mywr = csv.writer(testF, delimiter=' ')
        for label, s in enumerate(sets):  # label = index (0: FAKE, 1: REAL)
            setDir = sourceDir + "/" + s
            usersDir = os.listdir(setDir)  ## 50 users
            usersDir.sort()
            for i, user in enumerate(usersDir):
                # Test: users 14 to 50 + right eye of user 13 (i.e.: right of index 12 + index 13 to 49
                if i < 12:
                    continue
                files = os.listdir(setDir + "/" + user)
                for img in files:
                    if i == 12:
                        if "_r_" in img:
                            mywr.writerow([s + '/' + user + '/' + img] + [label])
                    else:
                        mywr.writerow([s + '/' + user + '/' + img] + [label])


#######################################################
def create_labelsTxtFile_Warsaw():
    categories_wasaw = ["PRNT", "REAL"]

    with open(categories_file, 'w') as writeF:
        for s in categories_wasaw:
            writeF.write("%s\n" % s)

    ###################
    subF = "Training"
    trainSubFolder = SET_rootFolder + "/PNG/" + subF
    trainF = trainFile
    with open(trainF, 'wb') as trainF:
        mywr = csv.writer(trainF, delimiter=' ')
        files = os.listdir(trainSubFolder)
        for img in files:
            if ("_" + categories_wasaw[1] + "_") in img:
                l = 1
            elif ("_" + categories_wasaw[0] + "_") in img:
                l = 0
            mywr.writerow([subF + '/' + img] + [l])

    ###################
    testF = testfile
    with open(testF, 'wb') as testF:
        mywr = csv.writer(testF, delimiter=' ')

        subF = "Testing"
        testSubFolder = SET_rootFolder + "/PNG/" + subF
        files = os.listdir(testSubFolder)
        for img in files:
            if "_REAL_" in img:
                l = 1
            elif "_PRNT_" in img:
                l = 0
            mywr.writerow([subF + '/' + img] + [l])

        subF = "Testing.Supplement"
        testSupSubFolder = SET_rootFolder + "/PNG/" + subF
        files = os.listdir(testSupSubFolder)
        for img in files:
            if "_REAL_" in img:
                l = 1
            elif "_PRNT_" in img:
                l = 0
            mywr.writerow([subF + '/' + img] + [l])


#######################################################
def create_labelsTxtFile_MobBioFake():
    with open(categories_file, 'w') as writeF:
        for s in sets:  # label = index (0: FAKE, 1: REAL)
            writeF.write("%s\n" % s)

    ###################
    subD = "Mobbiofake_DB_train"
    subF = SET_rootFolder + "/" + subD
    trainF = trainFile
    with open(trainF, 'wb') as trainF:
        mywr = csv.writer(trainF, delimiter=' ')
        for label, s in enumerate(sets):  # label = index (0: FAKE, 1: REAL)
            setDir = subF + "/" + s
            usersDir = os.listdir(setDir)  ## 50 users
            usersDir.sort()
            for user in usersDir:
                files = os.listdir(setDir + "/" + user)
                for img in files:
                    mywr.writerow([subD + '/' + s + '/' + user + '/' + img] + [label])

    ###################
    sets_test = ["fake", "real"]
    subD = "Mobbiofake_DB_test"
    subF = SET_rootFolder + "/" + subD
    testF = testfile
    with open(testF, 'wb') as testF:
        mywr = csv.writer(testF, delimiter=' ')
        for label, s in enumerate(sets_test):  # label = index (0: FAKE, 1: REAL)
            setDir = subF + "/" + s
            files = os.listdir(setDir)
            for img in files:
                mywr.writerow([subD + '/' + s + '/' + img] + [label])


#######################################################
def create_labelsTxtFile():
    if DB == 'ATVS':
        create_labelsTxtFile_ATVS()

    elif DB == 'Warsaw':
        create_labelsTxtFile_Warsaw()

    elif DB == 'MobBioFake':
        create_labelsTxtFile_MobBioFake()


#######################################################
create_labelsTxtFile()
