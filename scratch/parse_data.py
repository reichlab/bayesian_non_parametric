
avg_tmp = []
with open("avg_temp_san_juan.csv") as f:
	for line in f.readlines():
		try:
			avg_tmp.append(float(line.replace("\n","")))
		except:
			pass

avg_tmp_weekly = []

for i in range(0,len(avg_tmp),7):
	avg_tmp_weekly.append(sum(avg_tmp[i:i+7]))

print (avg_tmp_weekly[:988])