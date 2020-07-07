import os
print("\t\t\t HELPING HAND FOR FARMERS")
print("\t\t\t__________________________________")
while(True):
	print("""
		1-> PREDICT THE RAINFALL
		2-> PREDICT THE MANGO LEAF VARIETY
		3-> EXIT
      		""")
	print("""enter your choice """,end="")
	ch=input()
	if ch=="1":
		os.system("python C:/Users/Mon_Amour/Desktop/dataset_project/Rainfall_predictor.py")
	elif ch=="2":
		os.system("python C:/Users/Mon_Amour/Desktop/dataset_project/Mango_predictor.py")
	elif ch=="3":
		print("Thanku ")
		exit(0)
	else:
		print("Wrong input")