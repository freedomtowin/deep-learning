import datetime
import time
import importlib
import Dataset
import Bot
importlib.reload(Dataset)
importlib.reload(Bot)


def run_hour():
	start_time=time.time()
	end_time=time.time()
	while (start_time-end_time)<60*60:
	
		bot.execute()
		r = int(np.random.uniform(60,200))
		time.sleep(r)
		end_time=time.time()
		print()	
	bot.train()