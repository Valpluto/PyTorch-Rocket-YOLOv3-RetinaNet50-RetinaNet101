import torch
from rocketbase import Rocket
from PIL import Image
import os
import argparse
import pandas as pd
from timeit import default_timer as timer

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'number of images do you need')
args = ap.parse_args()



input_path = "images/train/train2014"
all_img = os.listdir(input_path)
retiananet_cat = []
yolov3_cat=[]
yolov3_time =[]
retinanet_time=[]
numberoftime = 0
csv_file_name = "output" 

for img in all_img:
	print("\n number of image - " + str(numberoftime) + "\n")

	if(numberoftime >= int(args.image)):
		break

	image_path = input_path + "/" + img    


	img = Image.open(image_path)

	# --- LOAD ROCKET ---
	# Select the Rocket you want to test
	rocket_ret = "igor/retinanet"
	# rocket = "igor/retinanet-resnet101-800px"
	rocket_yolo = "lucas/yolov3"

	start = timer()

	model = Rocket.land(rocket_ret).eval()

	# --- DETECTION ---
	print('Using the rocket to do object detection on \'' + image_path + '\'...')
	with torch.no_grad():
	    img_tensor = model.preprocess(img)
	    out = model(img_tensor)

	print('Object Detection successful! ')

	end = timer()

	retinanet_time.append((end-start))

	print(end-start)

	# # --- OUTPUT ---
	# # Print the output as a JSON
	bboxes_out = model.postprocess(out, img)
	# print(len(bboxes_out), 'different objects were detected:')
	# print(*bboxes_out, sep='\n')

	retiananet_cat.append(bboxes_out)

	# #Display the output over the image
	# img_out = model.postprocess(out, img, visualize=True)
	# img_out_path = 'out.jpg'
	# img_out.save(img_out_path)
	# print('You can see the detections on the image: \'' + img_out_path + '\'.')

	
	# YOLO 

	start = timer()

	model = Rocket.land(rocket_yolo).eval()

	# --- DETECTION ---
	print('Using the rocket to do object detection on \'' + image_path + '\'...')
	with torch.no_grad():
	    img_tensor = model.preprocess(img)
	    out = model(img_tensor)

	print('Object Detection successful! ')

	# --- OUTPUT ---
	# Print the output as a JSON
	bboxes_out = model.postprocess(out, img)
	#print(len(bboxes_out), 'different objects were detected:')
	#print(*bboxes_out, sep='\n')

	end = timer()

	yolov3_time.append((end-start))

	print(end-start)



	yolov3_cat.append(bboxes_out)

	# #Display the output over the image
	# img_out = model.postprocess(out, img, visualize=True)
	# img_out_path = 'out.jpg'
	# img_out.save(img_out_path)
	# print('You can see the detections on the image: \'' + img_out_path + '\'.')
	numberoftime+=1

	#if(numberoftime%2 ==0):
	df = pd.DataFrame(list(zip(all_img, retiananet_cat, retinanet_time, yolov3_cat, yolov3_time)), columns =['Image Name', 'retiananet_detected','r_time' ,'yolo_dected', 'y_time'])
	df.to_csv((csv_file_name + str(numberoftime) + '.csv'), sep=',')


df = pd.DataFrame(list(zip(all_img, retiananet_cat, retinanet_time,  yolov3_cat, yolov3_time)), columns =['Image Name', 'retiananet_detected','r_time' ,'yolo_dected', 'y_time'])
df.to_csv('output.csv', sep=',')
