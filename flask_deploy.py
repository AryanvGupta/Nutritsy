from flask import Flask , request , render_template ,Markup, url_for
import pickle
from tensorflow.keras import models
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

fruits = ['Apple', 'Apple Green', 'Apple Red', 
			'Banana', 'Blueberry', 
			'Carambola', 'Cauliflower', 
			'Guava', 
			'Kiwi', 
			'Mango', 'Muskmelon', 
			'Onion', 'Orange', 
			'Peach', 'Pear', 'Pepper Green', 'Persimmon', 'Pineapple', 'Pitaya', 'Plum', 'Pomegranate', 'Potato', 
			'Raspberry', 
			'Strawberry', 
			'Tomatoes', 
			'Watermelon']


app = Flask(__name__)
app.debug = True

@app.route("/")
def home():
	return render_template("home.html")

@app.route("/upload",methods = ["POST"])
def upload_image_page():
	return render_template("demo.html")


@app.route("/info" , methods = ["POST"])
def information_page():

	try:
		image_name = [i for i in request.form.values()][0]

		test = cv2.imread("test_images/" + image_name)
		test = np.array([cv2.resize(test, (70,70))])

		model = models.load_model("fruits_classification")
		predicted_index = np.argmax(model.predict([test])) 

		data = pd.read_excel("Fruits & Vegi Dataset.xlsx",header = 1)
		searched = fruits[predicted_index]
		fruit_ = data["Name"].tolist()
		fruit_ = [i.strip() for i in fruit_]
		index =fruit_.index(searched.strip())
		x = data.iloc[index]

		energy =  x["Energy (kcal)"]
		fats = x["Fat (g)"]
		protein = x["Protein (g)"]
		carbs = x["Carbohydrate (g)"]
		sugar = x["Sugars (g)"]
		fiber = x["Fibre (g)"]
		sodium = x["Sodium (mg)"]
		calcium = x["Calcium (mg)"]

		datapath = "static/displayImage/"

		img_src = "static/displayImage/area1.jpg"

		for i in os.listdir(datapath):
			if ( searched.lower().strip().replace(" ","_") in i.lower().strip() ) or (searched.lower().strip().replace(" ","_") == i.lower().strip() ) :
				img_src = os.path.join(datapath,i)
				break
            
		img_src = img_src.replace(" ","_")
		
		return render_template("out.html",Fruit_Name = searched, energy=energy, fats=fats, protein=protein, carbs=carbs, sugar=sugar, fiber=fiber, sodium=sodium, calcium=calcium, img_src = img_src  )

	except:
		return render_template("demo.html")


@app.route("/recommend" , methods = ["POST"])
def recommendation_page():
	dataset_file = open("dataset/recommend/dataset.pickle","rb")
	final_vector  = pickle.load(dataset_file)

	fruit_file = open("dataset/recommend/fruits.pickle","rb")
	fruits = pickle.load(fruit_file)

	for value in fruits:
		fruits[fruits.index(value)] = value.lower()
    	
	fruit_name = request.args.get('fruit_name')

	#if any values is not in pickle file then it will take 1 values by default i.e show recommendation of apple
	try:
		val = fruits.index(fruit_name.lower())

	except:
		val = 1

	no_of_recom = 4

	def rmd_model(dataset,fruit_index,recommendations):
		distance = pairwise_distances(dataset,dataset[fruit_index])
		rmd_index = np.argsort(distance.flatten())[1:recommendations + 1]
		return rmd_index

	rmd = rmd_model(final_vector,val,no_of_recom)
	fruit1 = fruits[rmd[0]].strip().replace(" ","_")
	fruit2 = fruits[rmd[1]].strip().replace(" ","_")
	fruit3 = fruits[rmd[2]].strip().replace(" ","_") 
	fruit4 = fruits[rmd[3]].strip().replace(" ","_")
	
	datapath = "static/displayImage/"

	f1_src = "static/displayImage/area1.jpg"
	f2_src = "static/displayImage/area1.jpg"
	f3_src = "static/displayImage/area1.jpg"
	f4_src = "static/displayImage/area1.jpg"



	for i in os.listdir(datapath):
		if ( fruit1.lower().strip() in i.lower().strip() ) or ( fruit1.lower().strip() == i.lower().strip() ):
			f1_src = os.path.join(datapath,i)
			f1_src = f1_src.replace(" ","_")
			break

	for i in os.listdir(datapath):
		if ( fruit2.lower().strip() in i.lower().strip() ) or ( fruit2.lower().strip() == i.lower().strip() ):
			f2_src = os.path.join(datapath,i)
			f2_src = f2_src.replace(" ","_")
			break

	for i in os.listdir(datapath):
		if ( fruit3.lower().strip() in i.lower().strip() ) or ( fruit3.lower().strip() == i.lower().strip() ):
			f3_src = os.path.join(datapath,i)
			f3_src = f3_src.replace(" ","_")
			break

	for i in os.listdir(datapath):
		if ( fruit4.lower().strip() in i.lower().strip() ) or ( fruit4.lower().strip() == i.lower().strip() ):
			f4_src = os.path.join(datapath,i)
			f4_src = f4_src.replace(" ","_")
			break


	f1_info = ""
	f2_info = ""
	f3_info = ""
	f4_info = ""

	with open(f"dataset/fruit_Info/{fruit1}",encoding="utf8") as f:
		f1_info = f.read()

	with open(f"dataset/fruit_Info/{fruit2}", encoding="utf8") as f:
		f2_info = f.read()

	with open(f"dataset/fruit_Info/{fruit3}" , encoding="utf8") as f:
		f3_info = f.read()

	with open(f"dataset/fruit_Info/{fruit4}" , encoding="utf8") as f:
		f4_info = f.read()

	fruit1 = fruit1.replace("_"," ").capitalize()
	fruit2 = fruit2.replace("_"," ").capitalize()
	fruit3 = fruit3.replace("_"," ").capitalize()
	fruit4 = fruit4.replace("_"," ").capitalize()

	return render_template("rec.html",Fruit1 = fruit1.capitalize(), Fruit2 = fruit2.capitalize(), Fruit3 =fruit3.capitalize() , Fruit4 =fruit4.capitalize() , f1_src = f1_src , f2_src = f2_src , f3_src = f3_src , f4_src = f4_src, f1_info = f1_info , f2_info = f2_info , f3_info = f3_info , f4_info = f4_info)


@app.route("/feature")
def feature_page():
	return render_template("fea.html")


@app.route("/developers")
def developers_page():
	return render_template("dev.html")


@app.route("/moreinfoEnglish", methods = ["POST"])
def moreinfo_pageEnglish():

	fruit_name = request.args.get('fruit_name').replace(" ","_")

	datapath = "static/displayImage/"

	img_src = "static/displayImage/area1.jpg"

	for i in os.listdir(datapath):
		if ( fruit_name.lower().strip().replace(" ","_") in i.lower().strip() ) or ( fruit_name.lower().strip().replace(" ","_") == i.lower().strip() ) :
			img_src = os.path.join(datapath,i)
			img_src = img_src.replace(" ","_")
			break


	with open(f"dataset/fruit_Info/{fruit_name}") as f:
		info = f.read()
		f.close()

	fruit_audio = f"static/fruit_audio/{fruit_name}.mp3"

	fruit_name = fruit_name.replace("_"," ")

	return render_template("info.html", info = info, img_src = img_src, fruit_name = fruit_name, fruit_audio = fruit_audio ,lang="Hindi")
	
	
@app.route("/moreinfoHindi", methods = ["POST"])
def moreinfo_pageHindi():

	fruit_name = request.args.get('fruit_name').replace(" ","_")

	datapath = "static/displayImage/"

	img_src = "static/displayImage/area1.jpg"

	for i in os.listdir(datapath):
		if ( fruit_name.lower().strip().replace(" ","_") in i.lower().strip() ) or ( fruit_name.lower().strip().replace(" ","_") == i.lower().strip() ):
			img_src = os.path.join(datapath,i)
			img_src = img_src.replace(" ","_")
			break

	with open(f"dataset/fruit_Info_hindi/{fruit_name}",encoding = "utf-8") as f:
		info = f.read()

	fruit_audio = f"static/fruit_audio_hindi/{fruit_name}.mp3"

	fruit_name = fruit_name.replace("_"," ")
	
	return render_template("info.html", info = info, img_src = img_src, fruit_name = fruit_name, fruit_audio = fruit_audio ,lang="English")
	

@app.route("/recommendinfo", methods = ["POST"])
def recommended_fruit_info():

	data = pd.read_excel("Fruits & Vegi Dataset.xlsx",header = 1)
	searched = request.args.get('fruitname').lower().strip()
	fruit_ = data["Name"].tolist()
	fruit_ = [i.strip().lower() for i in fruit_]
	index =fruit_.index(searched.strip())
	x = data.iloc[index]

	energy =  x["Energy (kcal)"]
	fats = x["Fat (g)"]
	protein = x["Protein (g)"]
	carbs = x["Carbohydrate (g)"]
	sugar = x["Sugars (g)"]
	fiber = x["Fibre (g)"]
	sodium = x["Sodium (mg)"]
	calcium = x["Calcium (mg)"]

	datapath = "static/displayImage/"

	img_src = "static/displayImage/area1.jpg"

	fruit_name = searched.replace(" ","_")
	for i in os.listdir(datapath):
		if ( fruit_name.lower().strip().replace(" ","_") in i.lower().strip() ) or ( fruit_name.lower().strip().replace(" ","_") == i.lower().strip() ):
			img_src = os.path.join(datapath,i)
			img_src = img_src.replace(" ","_")
			break
            
	img_src = img_src.replace(" ","_")
	searched = searched.capitalize()
		
	return render_template("out.html",Fruit_Name = searched, energy=energy, fats=fats, protein=protein, carbs=carbs, sugar=sugar, fiber=fiber, sodium=sodium, calcium=calcium, img_src = img_src  )



if __name__ == '__main__':
	app.run(debug = True)