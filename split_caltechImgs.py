# -*- coding: utf-8 *-

###################################################################################################
#  Copyright 2016 Hidenao Abe (hidenao@shonan.bunkyo.ac.jp)
#
# 本ソースコードは，Apache License Version 2.0（「本ライセンス」）に基づいてライセンスされます。
# あなたがこのファイルを使用するためには、本ライセンスに従わなければなりません。
# 本ライセンスのコピーは下記の場所から入手できます。
# http://www.apache.org/licenses/LICENSE-2.0
#
# 適用される法律または書面での同意によって命じられない限り、
# 本ライセンスに基づいて頒布されるソフトウェアは、明示黙示を問わず、
# いかなる保証も条件もなしに「現状のまま」頒布されます。
#
# 本ライセンスでの権利と制限を規定した文言については、本ライセンスを参照してください。
#（この日本語訳は，http://sourceforge.jp/projects/opensource/wiki/licenses%2FApache_License_2.0によるものです）
#
###################################################################################################

import os
import random
import shutil
import csv
import cv2
import numpy as np
from pprint import pprint
from BagOfFeatures import * 

img_dir = "101_ObjectCategories"
lowerLimit = 20
upperLimit = 40
numFolds = 10
temp_dir = "test"

vw_method = "ORB" # bag of visual wordsを求める際に利用する特徴点抽出手法
numVW = 256
dataset_dir = "dataset" #ファイル名は"(train|test)_フォールド数_クラス名Byクラス名.csv"など

#大域変数
folds_list=list()
labels = list() #ラベル候補
train_file = ""
test_file=""

def devideFolds():
	# イメージディレクトリ以下のサブディレクトリ（クラスラベル）を取得
	subdirs = os.listdir(img_dir)

	numfiles = 0;
	img_files = dict();
	num_img_files = dict();

	for subdir in subdirs:
		files = os.listdir(img_dir + "/" + subdir);
		if len(files) >= lowerLimit and len(files)<=upperLimit:
			labels.append(subdir)
			img_files[subdir]=files
			numfiles += len(files)
			num_img_files[subdir] = len(files)
	
	#クラスラベル毎でシャッフルした訓練(train_n)と検証(test_n)にファイルを分ける
	for subdir in img_files:
		files = img_files[subdir]
		random.shuffle(files)

		count = 0
		for file in files: #検証用データに”フォールド数分の1”のファイルをコピー
			num = int(count) % int(numFolds)
			if not num in folds_list:
				folds_list.append(num)
			if not os.path.exists(temp_dir + "/train_"+ str(num)):
				os.mkdir(temp_dir + "/train_"+ str(num))
			if not os.path.exists(temp_dir + "/train_"+ str(num) + "/"+subdir):
				os.mkdir(temp_dir + "/train_"+ str(num) + "/"+subdir)
			if not os.path.exists(temp_dir + "/test_"+ str(num)):
				os.mkdir(temp_dir + "/test_"+ str(num))
			if not os.path.exists(temp_dir + "/test_"+ str(num) + "/"+subdir):
				os.mkdir(temp_dir + "/test_"+ str(num) + "/"+subdir)

			src_file = img_dir+"/"+subdir+"/"+file
			temp_file = temp_dir+"/test_"+ str(num)+"/"+subdir+"/"+file
			shutil.copyfile(src_file, temp_file)

			count+=1

		#訓練用データとして検証データと重ならない”フォールド数分の(フォールド数-1)”のファイルをコピー
		for fold in folds_list: 
			for file in files:
				src_file = img_dir+"/"+subdir+"/"+file
				temp_file = temp_dir+"/train_"+ str(fold)+"/"+subdir+"/"+file
				if not os.path.exists(temp_dir+"/test_"+ str(fold)+"/"+subdir+"/"+file) :
					shutil.copyfile(src_file, temp_file)
					#print("OK")

def makeDatasetOneByOne():
	
	for fold in folds_list:
		checked = list()
		for label1 in labels:
			checked.append(label1)

			for label2 in labels:
				if label1 != label2 and label2 not in checked:
					class_labels = list()
					class_labels.append(label1)
					class_labels.append(label2)
					train_path = temp_dir+"/train_"+str(fold)+"/"
					global train_file
					train_file = dataset_dir+"/train_"+str(fold)+"_"+"By".join(class_labels)+".csv"
					test_path = temp_dir+"/test_"+str(fold)+"/"
					global test_file
					test_file = dataset_dir+"/test_"+str(fold)+"_"+"By".join(class_labels)+".csv"
					makeDataset(train_path, test_path, class_labels)

def makeDataset(train_path, test_path, class_labels):

	print("Making training dataset with bag of visual words.")
	global train_file
	global test_file
	
	# クラスラベルによって別々のディレクトリに振り分けられている画像を読み込む
	train_features={}
	for class_label in class_labels:

		if class_label == "vsOthers":
			t_list = os.path.listdir(train_path)
			for t_label in t_list:
				train_images={}
				if t_label in class_labels:
					continue
				else:
					train_images[t_label]=loadImages(train_path+"/"+t_label)
					train_features[t_label] = extractFeatures(train_images[t_label],method=vw_method)
					if "train_all" in train_feature:
						train_features["train_all"] = np.vstack(np.append(train_features["train_all"],train_features[t_label]))
					else:
						train_features["train_all"] = train_features[t_label]

		else:
			train_images={}
			train_images[class_label]=loadImages(train_path+"/"+class_label)
			train_features[class_label] = extractFeatures(train_images[class_label],method=vw_method)

			if "train_all" in train_features:
				train_features["train_all"] = np.vstack(np.append(train_features["train_all"],train_features[class_label]))
			else:
				train_features["train_all"] = train_features[class_label]

	#各画像に対応する記述子からヒストグラムを作成
        #コードブック数をnumVWとして、BoFを計算
	bof=BagOfFeaturesGMM(codebookSize=numVW)
	bof.train(train_features["train_all"])

	#訓練データをVoVW表現に変換（各クラスラベルごと）
	train_hist={}
	for class_label in class_labels:

		if class_label == "vsOthers":
			t_list = os.path.listdir(train_path)
			for t_label in t_list:
				if t_label in class_labels:
					continue
				else:
					train_hist[t_label] = map(lambda a:bof.makeHistogram(np.matrix(a)),train_features[t_label])

		else:
			train_hist[class_label] = map(lambda a:bof.makeHistogram(np.matrix(a)),train_features[class_label])

	#train_histにクラスラベルを付けてファイルに保存
	header = list()
	for num in range(0,numVW):
		attName = "VW" + str(num)
		header.append(attName)
	saveCSVHeader(train_file,header,"Class")

	for class_label in class_labels:

		if class_label == "vsOthers":
			t_list = os.path.listdir(train_path)
			for t_label in t_list:
				if t_label in class_labels:
					continue
				else:

					appendMatrix2CSV(train_file, train_hist[t_label], class_label);
		else:
			appendMatrix2CSV(train_file, train_hist[class_label], class_label);

	#検証用データの特徴点を抽出
	test_features={}
	for class_label in class_labels:
		if class_label == "vsOthers":
			t_list = os.path.listdir(test_path)
			for t_label in t_list:
				train_images={}
				if t_label in class_labels:
					continue

				else:
					test_images[t_label]=loadImages(test_path+"/"+t_label)
					test_features[t_label] = extractFeatures(test_images[t_label],method=vw_method)
		else:
			test_images={}
			test_images[class_label]=loadImages(test_path+"/"+class_label)
			test_features[class_label] = extractFeatures(test_images[class_label],method=vw_method)

		
	#検証用データをVoVW表現に変換（各クラスラベルごと）
	test_hist={}
	for class_label in class_labels:

		if class_label == "vsOthers":
			t_list = os.path.listdir(test_path)
			for t_label in t_list:
				if t_label in class_labels:
					continue

				else:
					test_hist[t_label] = map(lambda a:bof.makeHistogram(np.matrix(a)),test_features[t_label])

		else:
			test_hist[class_label] = map(lambda a:bof.makeHistogram(np.matrix(a)),test_features[class_label])

	# 検証用データにクラスラベルを付けてファイルに保存
	header = list()
	for num in range(0,numVW):
		attName = "VW" + str(num)
		header.append(attName)
	saveCSVHeader(test_file,header,"Class")

	for class_label in class_labels:

		if class_label == "vsOthers":
			t_list = os.path.listdir(test_path)
			for t_label in t_list:
				if t_label in class_labels:
					continue
				else:
					appendMatrix2CSV(test_file, test_hist[t_label], class_label);
		else:
			appendMatrix2CSV(test_file, test_hist[class_label], class_label);
# end of makeDataset(train_path, test_path, class_labels):

def loadImages(path): #指定されたフォルダから画像を読み出す関数
	import os
	imagePathes=map(lambda a:os.path.join(path,a),os.listdir(path))
	images=map(cv2.imread,imagePathes)
	return(images)

def extractFeatures(images,method):
	if method == "ORB":
		detector = cv2.ORB_create()
	elif method == "KAZE":
		detector = cv2.KAZE_create()
	elif method == "AKAZE":
		detector = cv2.AKAZE_create()
	elif method == "BRISK":
		detector = cv2.BRISK_create()
	elif method == "FAST":
		detector = cv2.FastFeatureDetector_create() # FAST特徴点検出法
	else:
		print "手法がORB, KAZE, AKAZE, BRISK，FAST以外なのでFASTによる特徴点抽出を行います．\n"
		detector = cv2.FastFeatureDetector_create() # FAST特徴点検出法（その他の場合でも）

	keypoints = map(detector.detect, images) #keypointsは各画像の特徴点の配列の配列となる
	descriptors = map(lambda a,b:detector.compute(a,b)[1], images,keypoints) #descriptorsは各画像の特徴点がどこにあるか（記述子）の配列の配列となる
	return(descriptors)

#ファイルに1行書き込む
def saveCSVHeader(filename, line, append_str):

	line.append(append_str)

	with open(filename, 'w') as f:
		writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
		writer.writerow(line)     # list（1次元配列）の場合


def appendMatrix2CSV(filename, np_matrix, append_str):

	with open(filename, 'a') as f:
		writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
		for np_list in np_matrix:
			w_list = np.append(np_list,append_str)
			pprint(w_list)
			writer.writerow(w_list)


if __name__== "__main__":

	# numFoldsに指定したn-fold cross validation用に訓練データと検証データを分割
	devideFolds()

	# クラスラベル1対1のデータセットを作成する
	makeDatasetOneByOne()




