from time import sleep
from getch import getch
from os import remove

src_dir = "/home/pi/final_project"
dst_dir_input = f"{src_dir}/imgs/Input"
dst_dir_output = f"{src_dir}/imgs/Output"


def generate_img_name():
	import datetime
	
	d = datetime.datetime.now()
	uuid = d.strftime("%d-%m-%y-%H-%M-%S")
		
	return f"IMG-{uuid}"


def take_photo(img_name):
	from glob import iglob
	from shutil import move
	from os import system, path
	
	
	# take a photo	
	system('libcamera-hello --qt-preview')
	system(f"libcamera-jpeg --ev 0.5 -t 500 -o {img_name}.jpg -n") 
	
	# move the picture from src to imgs/Input folder
	for jpgfile in iglob(path.join(src_dir, "*.jpg")):
		move(jpgfile, dst_dir_input)
	
def open_photo(img_name):
	from PIL import Image
	import matplotlib.pyplot as plt
	
	# show the result image
	img = Image.open(f"{dst_dir_output}/{img_name}.jpg")
	img.show()
	
	sleep(2)
	img.fp.close()


	
while True:  
	inp = getch() # Get  char - type : <str>
	
	if inp.lower() == "w":
		sleep(2) #to give time to take a picture
		
		img_name = generate_img_name()
		
		take_photo(img_name)
		
		# compute emotion
		exec(open("algos.py").read())
		
		open_photo(img_name)
		
		# remove photo
		remove(f"{dst_dir_input}/{img_name}.jpg")
		
		
		

